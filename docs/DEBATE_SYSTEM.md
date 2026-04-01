# Bull-Bear Adversarial Debate — 多空对抗性辩论系统

## 为什么单 Agent 分析有问题？

RD-Agent 原版用**同一个 LLM** 来生成假设和分析结果。这会导致**确认偏差（confirmation bias）**：

```
AI 生成假设: "动量因子在牛市中应该表现好"
   ↓
AI 分析结果: "虽然 IC 只有 0.01，但还是有改进空间，建议继续"
   ↓
AI 又分析: "IC 从 0.01 降到 0.005，但波动率降低了，建议继续"
   ↓
...10 轮后，AI 还在说"继续"
```

**问题**：AI 倾向于为自己之前的决策辩护（和人类一样），很少主动说"我之前的方向是错的"。

---

## 解决方案：让两个 AI 吵架

受 [TradingAgents](https://arxiv.org/abs/2412.20138)（2024）启发，我们引入对抗性辩论：

```
         ┌──────────────────────────┐
         │   回测结果 + 假设描述     │
         │   IC=0.01, ICIR=0.05    │
         └────────┬─────────────────┘
                  │
          ┌───────┴───────┐
          ▼               ▼
   ┌─────────────┐ ┌─────────────┐
   │  Bull Agent  │ │  Bear Agent  │
   │  (看多分析师) │ │  (看空分析师) │
   │              │ │              │
   │ "IC 虽然低   │ │ "IC 只有 0.01│
   │  但趋势向上  │ │  远低于阈值   │
   │  建议继续"   │ │  3轮没改善    │
   │              │ │  建议换方向"  │
   └──────┬──────┘ └──────┬──────┘
          │               │
          └───────┬───────┘
                  ▼
         ┌───────────────┐
         │  Judge Agent   │
         │  (裁判)        │
         │               │
         │ 综合两方论点   │
         │ 输出: PIVOT    │
         │ 置信度: 0.75   │
         └───────────────┘
```

### 三个角色

| 角色 | 立场 | 倾向 | 建议范围 |
|---|---|---|---|
| **Bull** | 看多 | 发现改进空间 | refine, continue, adjust |
| **Bear** | 看空 | 质疑根本假设 | pivot, abandon, switch |
| **Judge** | 中立 | 综合判断 | CONTINUE / PIVOT / NEUTRAL |

---

## Prompt 设计

### Bull Agent（看多分析师）

```
你是一个资深看多分析师。你的工作是发现当前研究方向的潜力和改进空间。

即使指标暂时不好，你也会寻找：
- 趋势是否在改善
- 是否有调参空间
- 假设本身是否有价值只是实现不好

你倾向于建议"继续优化"而不是"放弃"。
```

### Bear Agent（看空分析师）

```
你是一个资深看空分析师。你的工作是挑战当前研究方向的根本假设。

你会关注：
- 指标是否真的在改善，还是只是噪音
- 这个方向继续下去的机会成本
- 是否应该彻底换一个假设方向

你倾向于质疑和挑战，不轻易说"继续"。
```

### Judge（裁判）

```
基于以下多空两方分析，做出最终决策：

看多论点: {bull_argument}
看空论点: {bear_argument}

请输出：
1. 最终裁决 (CONTINUE / PIVOT / NEUTRAL)
2. 置信度 (0.0 - 1.0)
3. 具体下一步行动建议
```

---

## 成本分析

| 组件 | LLM 调用次数/轮 | 模型 | 成本/轮 |
|---|---|---|---|
| Bull Agent | 1 | DeepSeek Reasoner | ~$0.003 |
| Bear Agent | 1 | DeepSeek Reasoner | ~$0.003 |
| Judge | 1 | DeepSeek Chat | ~$0.001 |
| **总计** | **3** | — | **~$0.007** |

30 轮总成本：约 $0.21 — 相比整个回测费用（$1.5-3.0）只增加了约 10%。

---

## 使用方式

```python
from src.debate_agents import DebateAnalyzer

analyzer = DebateAnalyzer(
    debate_model="deepseek/deepseek-reasoner",
    judge_model="deepseek/deepseek-chat",
)

result = analyzer.debate(
    hypothesis="基于 20 日价格动量的多头因子",
    code="df['close'].pct_change(20)",
    metrics={"IC": 0.01, "ICIR": 0.05, "annual_return": 0.03},
    iteration=5,
)

print(result.verdict)        # Verdict.PIVOT
print(result.confidence)     # 0.75
print(result.bull_argument)  # "虽然 IC 低但..."
print(result.bear_argument)  # "5轮了还是这个水平..."
print(result.synthesis)      # "综合来看应该换方向"
```

---

## 裁决类型

| 裁决 | 含义 | 后续动作 |
|---|---|---|
| **CONTINUE** | 当前方向有潜力 | 继续优化当前假设 |
| **PIVOT** | 应该换方向 | 生成全新假设 |
| **NEUTRAL** | 信号不明确 | 微调后再看一轮 |

---

## 预期效果

| 指标 | 无辩论 | 有辩论 | 原因 |
|---|---|---|---|
| 无效迭代比例 | ~40% | ~15% | 更早发现死胡同 |
| 方向切换速度 | 平均 8 轮 | 平均 3 轮 | Bear 会主动建议换方向 |
| 最终因子质量 | 依赖运气 | 更稳定 | 减少确认偏差 |

辩论系统的价值不在于"做对"，而在于**更快地发现错误方向并止损**。
