# Cost Analysis: Multi-Model Routing Economics

## Pricing Reference (as of March 2026)

| Model | Input (miss) | Input (hit) | Output | Context |
|:---|---:|---:|---:|---:|
| DeepSeek V3.2 Chat | $0.28/M | $0.028/M | $0.42/M | 128K |
| DeepSeek V3.2 Reasoner | $0.28/M | $0.028/M | $0.42/M | 128K |
| Claude Sonnet 4.5 | $3.00/M | $0.30/M | $15.00/M | 200K |
| GPT-4o | $2.50/M | $1.25/M | $10.00/M | 128K |
| GPT-4o-mini | $0.15/M | $0.075/M | $0.60/M | 128K |

## Cost Model

### Per-Call Cost Formula

```
cost = input_tokens × ((1 - cache_hit_rate) × price_miss + cache_hit_rate × price_hit)
     + output_tokens × price_output
```

### Cache Hit Rate Estimation

DeepSeek V3.2 uses automatic prefix caching. In RD-Agent's iterative pipeline:

- **First call of a session**: 0% cache hit (cold start)
- **Subsequent calls within same module**: ~70–80% cache hit (shared system prompt + history prefix)
- **Cross-module calls**: ~30–50% cache hit (shared system prompt only)
- **Weighted average across 30 iterations**: ~50–60% cache hit rate

### Per-Iteration Token Budget

Based on paper settings (Appendix C.1) and prompt analysis (Appendix E):

| Module | Calls | Avg Input/call | Avg Output/call | Total Input | Total Output |
|:---|---:|---:|---:|---:|---:|
| Specification | 0.5 | 2,000 | 500 | 1,000 | 250 |
| Synthesis | 1.5 | 10,000 | 1,000 | 15,000 | 1,500 |
| Implementation | 13 | 3,000 | 800 | 39,000 | 10,400 |
| Analysis | 1.5 | 4,000 | 800 | 6,000 | 1,200 |
| **Total/iter** | **~16.5** | | | **~61,000** | **~13,350** |

## Scenario Comparison (30 iterations)

### Scenario A: All GPT-4o (Paper Baseline)

```
Input:  30 × 61K × $2.50/M = $4.575
Output: 30 × 13.35K × $10.00/M = $4.005
Total ≈ $8.58 (consistent with paper's "<$10" claim)
```

### Scenario B: All DeepSeek Chat (Budget)

```
Input:  30 × 61K × (0.5 × $0.028 + 0.5 × $0.28)/M = $0.281
Output: 30 × 13.35K × $0.42/M = $0.168
Total ≈ $0.45
```

### Scenario C: Multi-Model Routing (Optimized)

Synthesis + Analysis use Reasoner; Implementation uses Chat. Both are DeepSeek V3.2 with identical pricing, but Reasoner activates chain-of-thought for higher quality.

```
Strong (Synthesis+Analysis):
  Input:  30 × 21K × (0.5 × $0.028 + 0.5 × $0.28)/M = $0.097
  Output: 30 × 2.7K × $0.42/M = $0.034

Efficient (Implementation+Spec):
  Input:  30 × 40K × (0.5 × $0.028 + 0.5 × $0.28)/M = $0.185
  Output: 30 × 10.65K × $0.42/M = $0.134

Total ≈ $0.45
```

Note: Since DeepSeek V3.2 uses unified pricing for both chat and reasoner modes, the cost is identical to Scenario B. The benefit is purely in quality, not cost. The cost advantage of routing only appears when mixing providers (e.g., Claude for Synthesis + DeepSeek for Implementation).

### Scenario D: Premium Routing (Claude + DeepSeek)

```
Strong (Claude Sonnet for Synthesis+Analysis):
  Input:  30 × 21K × (0.5 × $0.30 + 0.5 × $3.00)/M = $1.039
  Output: 30 × 2.7K × $15.00/M = $1.215

Efficient (DeepSeek Chat for Implementation+Spec):
  Input:  30 × 40K × (0.5 × $0.028 + 0.5 × $0.28)/M = $0.185
  Output: 30 × 10.65K × $0.42/M = $0.134

Total ≈ $2.57
```

## Summary Table

| Scenario | Total Cost | vs. Paper | Quality Expectation |
|:---|---:|---:|:---|
| A: All GPT-4o (paper) | ~$8.58 | 1.0× | Verified in paper |
| B: All DeepSeek Chat | ~$0.45 | 0.05× | Good (needs validation) |
| C: DeepSeek Reasoner + Chat | ~$0.45 | 0.05× | Better than B (quality gain, same cost) |
| D: Claude + DeepSeek | ~$2.57 | 0.30× | Best expected (needs validation) |

## Key Takeaway

The most impactful optimization is **switching from GPT-4o to DeepSeek V3.2** (85–95% cost reduction). Multi-model routing provides an additional quality boost at the same cost by using the Reasoner mode for high-impact stages, or enables mixing in Claude for maximum quality at still 70% less than the paper baseline.
