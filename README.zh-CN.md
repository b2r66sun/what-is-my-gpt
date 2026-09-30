# wimgpt — 你聊的到底是什么模型？

[English](README.md)

通过探测训练数据截止日期推断实际服务你的是哪个模型，判断gpt降智：把一组带日期的真实
事件题发给模型，回复贴回来，得到各候选模型的后验分布。

![示例输出](assets/panel.png)


## 使用

1. 通过 http 打开 `index.html`
2. 复制提问，在 ChatGPT 新开对话（关闭联网/搜索）发送
3. 粘贴回复，点 Analyze（Load sample → Analyze 可看演示）

抽样：每次问卷从每条截止缝隙抽 N 题（金丝雀必含）；题号沿用题库全局
编号，任意子集的回复都能对回题目。分析后可点「Round-2 quiz」围绕 MAP
所在带重新抽样做定向加测。当前抽样（seed）跨刷新保留。

## 题库维护

`tools/gen_bank.py` 是确定性层：抓取 Wikipedia 时事日页并做**无结构
假设的压平**（标签级转换），连同文章摘要生成 `raw-context.md` 供日期
交叉核对。条目提取、措辞、关键词和可猜性判断是 agent 任务
（`tools/agent-task.md`）——用 GitHub Agentic Workflows（[gh-aw](https://github.com/github/gh-aw)，
消耗 Copilot premium requests，公开仓库 Actions 计算免费）或任意 agent
CLI 执行。`--merge` 写回产出；`--canaries` 生成虚构控制题；
`--check`/`--coverage` 校验。

CI（`.github/workflows/bank.yml`）：`check` 在 push 时运行；`raw` 每周一
03:00 UTC（或手动触发）生成上下文文件并附两个金丝雀开 PR——agent
执行位在 workflow 内注释说明。

## 原理

### 形式化

问卷为 $n$ 道带日期的事件题 $q_1,\dots,q_n$，事件日期 $d_1 \le \dots \le d_n$；
候选模型 $m_1,\dots,m_M$，训练数据截止日 $c_1,\dots,c_M$，先验
$\pi_j = P(m_j)$，$\sum_j \pi_j = 1$（默认均匀）。每题观测
$r_i \in \{0,1\}$：1 = 知道该事件。

### 似然模型

$$P(r_i = 1 \mid m_j) = \begin{cases} p, & d_i \le c_j \\[2pt] \varepsilon, & d_i > c_j \end{cases}$$

$p = 0.9$（窗口内的事件可能回忆失败）；$\varepsilon = 0.05$（截止后的事件
可能被猜中或幻觉；若 $\varepsilon = 0$，一次截止后的命中会将该候选的似然
归零，命中在所有候选截止日之后则全部归零）。

$$L_j = P(r_{1:n} \mid m_j) = \prod_{i=1}^{n} P(r_i \mid m_j)$$

### 充分统计量

```math
n_j = \#\{i : d_i \le c_j\}, \qquad K_j = \sum_{d_i \le c_j} r_i, \qquad G_j = \sum_{d_i > c_j} r_i
```
$$L_j = p^{K_j}\,(1-p)^{\,n_j - K_j}\;\cdot\;\varepsilon^{G_j}\,(1-\varepsilon)^{\,n - n_j - G_j}$$

答案向量只通过 $(n_j, K_j, G_j)$ 进入推断：测试估计的是一个变点——按日期
排序的答案序列从 1 翻到 0 的位置。

### 后验

$$P(m_j \mid r) = \frac{\pi_j\, L_j}{\sum_{j'=1}^{M} \pi_{j'}\, L_{j'}}, \qquad \log \frac{P(m_a \mid r)}{P(m_b \mid r)} = \log \frac{\pi_a}{\pi_b} + \log \frac{L_a}{L_b}$$

- 均匀先验 ⇒ 后验 ∝ 似然
- 同截止日 ⇒ 对任何回答模式似然相同，后验比等于先验比（均匀先验下为
  50/50）

### 一道分界题的证据

日期落在 $c_a < d_i \le c_b$ 的题对 $b$ vs $a$ 的对数几率贡献：

$$r_i = 1:\ \ \log\frac{p}{\varepsilon} \approx 2.89 \text{ nats} \quad (\approx 18\text{ 倍})$$

$$r_i = 0:\ \ \log\frac{1-\varepsilon}{1-p} \approx 2.25 \text{ nats，指向 } a \quad (\approx 9.5\text{ 倍})$$

单题可能失误（窗口内 10% miss 率），每条截止缝隙放 2–3 道题。

### 数值示例

两个候选：$A$（截止日在 $d_2$ 前）、$B$（在后）。两题都在 $B$ 的窗口内，
只有 $q_1$ 在 $A$ 的窗口内。回复 $r = (1, 0)$：

$$L_A = 0.9 \cdot 0.95 = 0.855 \qquad L_B = 0.9 \cdot 0.10 = 0.09$$

- 均匀先验：$P(A \mid r) = 0.855 / (0.855 + 0.09) = 90.5\%$

### 缺失回答

从似然中整体省略（记 0 会偏向更早的截止日）。实答题数低于 60% 判定
run 无效。

### 输出

各模型后验；按截止带聚合的后验；MAP；后验熵 $H = -\sum_j P_j \log P_j$；
金丝雀检查（虚构事件——声称知道即触发幻觉警报）。非 JSON 回复按编号行
解析并用关键词判分。

## 数据文件

`questions.json`——带日期的事件（`id`、`date`、`question`、`truth`、
`keywords`、可选 `canary`）。事件放在各候选截止日的缝隙里；只用不可猜的
专名——答案可凭事前知识猜出的题，删除或改写

`models.json`——候选（`id`、`cutoff`），先验默认均匀。截止日为 2026-09
时点（部分为第三方转述，以 developers.openai.com 为准）：

| 模型 | 截止日 |
|---|---|
| gpt-6-astra | 2026-04-30 |
| gpt-6-sol | 2026-04-20 |
| gpt-6-luna | 2026-05-18 |
| gpt-5.6 | 2026-02-16 |
| gpt-5.5-pro / 5.5-mini | 2025-12-01 |
| gpt-5.4 / 5.2 | 2025-08-31 |
| gpt-5 / 5.1 | 2024-09-30 |
| gpt-5-mini / nano | 2024-05-31 |
| gpt-4.1 / o3 | 2024-06-01 |
| gpt-4o / 4o-mini | 2023-10-01 |

## 局限

- 测的是截止日；同截止日候选无法被区分
- 全局 $p$/$\varepsilon$：若题库残余可猜性超过 $\varepsilon=0.05$，每题
  Bayes 因子被高估——用 $\varepsilon \times 3$ 复跑检验稳健性
- 联网搜索使结果塌缩到最新截止日；新开对话、关闭搜索。自报可能被网页端
  系统提示压制——意外结果用自由回忆题复核
- 单对话的题目间有轻微上下文泄漏；拒答压低 $p$
- 官方可能静默更新 cutoff

## 数据来源

- 模型 cutoff 谱系：[llm-knowledge-cutoff-dates](https://github.com/HaoooWang/llm-knowledge-cutoff-dates)、
  [GPT-6 Astra 发布](https://openai.com/index/gpt-6-astra/)、
  [GPT-5.5 Pro cutoff](https://tutorsbot.com)、
  [GPT-5.6 cutoff](https://www.eesel.ai)、
  [GPT-6 Astra cutoff](https://dev.to)
- 事件日期：Wikipedia 时事门户
  [2025-10](https://en.wikipedia.org/wiki/Portal:Current_events/October_2025)、
  [2026-02](https://en.wikipedia.org/wiki/Portal:Current_events/February_2026)、
  [2026-03](https://en.wikipedia.org/wiki/Portal:Current_events/March_2026)
- 降级路由场景：[社区报告：5.6 Pro 被自动路由到 5.5 mini](https://community.openai.com)
