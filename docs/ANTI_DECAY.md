# Anti-Alpha-Decay Filter — 抗因子衰减过滤器

## 什么是因子衰减（Alpha Decay）？

简单说：**一个赚钱的策略，用的人多了就不赚钱了。**

更准确地说：
1. 你发现了一个有效的因子（比如"过去 20 天涨幅最大的股票"）
2. 它在回测中表现很好
3. 越来越多人发现了同样的因子
4. 大家都按这个信号交易 → 信号被"消费"掉了
5. 原来赚钱的策略慢慢失效

在 RD-Agent 的自动化循环中，类似的问题以另一种形式出现：
- AI 每轮生成新因子
- 但 AI 倾向于生成"安全"的、和之前类似的因子
- 30 轮下来，可能 70% 的因子本质上是同一个东西
- 这就是**因子拥挤（factor crowding）**

---

## AlphaAgent 论文的三重机制

我们的实现基于 [AlphaAgent](https://arxiv.org/abs/2502.16789)（KDD 2025 最佳论文之一），它提出了三层防线：

### 第一层：AST 结构相似度检查

**做什么**：把代码转换成抽象语法树（AST），比较结构是否雷同。

**为什么不用文本比较**：改个变量名就能骗过文本比较，但 AST 比较只看结构。

```python
# 这两段代码文本不同，但 AST 结构相同（只是变量名不同）
# Factor A
factor = df['close'].pct_change(20)

# Factor B（本质一样，只改了变量名）
result = df['close'].pct_change(20)

# AST 比较结果：相似度 = 1.0 → 拒绝 Factor B
```

**阈值**：相似度 > 85% → 拒绝（论文推荐值）

### 第二层：复杂度约束

**做什么**：限制因子公式的嵌套深度。

**为什么**：过于复杂的因子往往是过拟合——它在历史数据上表现很好，但在新数据上完全失效。

```python
# 深度 2（OK）
np.log(df['close'].pct_change(20))

# 深度 5（超标，大概率过拟合）
np.tanh(np.log(df['close'].rolling(20).mean() / df['close'].rolling(60).std()))
```

**阈值**：嵌套深度 > 5 → 拒绝

### 第三层：假设-代码对齐验证

**做什么**：用 LLM 检查"生成的代码是否真的实现了假设"。

**为什么**：AI 有时候会"偷懒"——假设说的是"动量因子"，但代码其实计算的是波动率。

```
假设: "基于过去20天的价格动量，看涨趋势的股票应该继续上涨"
代码: df['close'].rolling(20).std()  ← 这是波动率，不是动量！

LLM 判定: 对齐分数 = 0.2 → 拒绝
```

---

## 本项目的实现

### 文件
`src/alpha_filter.py`

### 类结构

```
AlphaDecayFilter（组合过滤器）
  ├── ASTSimilarityChecker   — Jaccard 相似度 on AST tokens
  ├── ComplexityChecker       — 递归计算嵌套深度
  └── AlignmentChecker        — LLM 验证假设-代码一致性
```

### 使用示例

```python
from src.alpha_filter import AlphaDecayFilter

filter = AlphaDecayFilter(
    similarity_threshold=0.85,  # 相似度阈值
    max_complexity_depth=5,      # 最大嵌套深度
    min_alignment_score=0.6,     # 最低对齐分数
)

# 注册已有因子
filter.add_existing_factor("momentum_20d", "df['close'].pct_change(20)")
filter.add_existing_factor("volatility_20d", "df['close'].rolling(20).std()")

# 检查新因子
result = filter.evaluate(
    code="df['close'].pct_change(20)",  # 和 momentum_20d 一样！
    hypothesis="A new momentum factor",
    check_alignment=False,
)

print(result.passed)           # False — 被相似度检查拒绝
print(result.rejection_reasons) # ["Too similar to 'momentum_20d' (similarity=1.00)"]
```

### 参数选择

| 参数 | 值 | 来源 | 调整建议 |
|---|---|---|---|
| similarity_threshold | 0.85 | AlphaAgent 论文 | 降到 0.7 会更严格，可能误杀 |
| max_complexity_depth | 5 | 经验值 | 量化策略通常 3-4 层就够 |
| min_alignment_score | 0.6 | 经验值 | 提高到 0.8 会更严格 |

### 成本

- AST 检查 + 复杂度检查：**免费**（纯 Python 计算）
- 对齐检查：1 次 LLM 调用（约 $0.001）
- 默认关闭对齐检查以节省成本，仅 AST + 复杂度就能过滤 60-80% 的重复因子

---

## 预期效果

| 指标 | 无过滤器 | 有过滤器 | 改善 |
|---|---|---|---|
| 因子去重率 | 0% | ~60-80% | 减少无效迭代 |
| 有效因子比例 | ~30% | ~70% | 更多有意义的因子 |
| IC 衰减速度 | 快 | 慢 | 因子存活期更长 |
