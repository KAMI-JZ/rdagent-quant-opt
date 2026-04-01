# Data Sources — 数据源说明

## 为什么不能只用 Yahoo Finance？

Yahoo Finance 免费且无需注册，但有严重缺陷：

| 问题 | 影响 | 严重程度 |
|---|---|---|
| **生存者偏差** | 只有"现在还活着"的股票，退市的没了 | 致命 |
| **拆分调整不完整** | 部分历史拆分未被正确调整 | 严重 |
| **数据延迟** | 免费数据通常延迟 15-20 分钟 | 中等 |
| **无 API 保障** | 随时可能被限速或关闭 | 中等 |

### 生存者偏差是什么？

假设你在 2008 年的投资组合里有 500 只股票。到 2024 年，有些公司倒闭了（比如雷曼兄弟），有些被收购了。如果你只用 2024 年还存在的股票去回测 2008 年的策略，你就跳过了所有失败的公司，结果看起来比实际好很多。

**这不是小问题**——学术研究表明，生存者偏差可以让回测收益率虚高 1-3% 每年。

---

## Polygon.io 的数据质量优势

| 特性 | Yahoo Finance | Polygon.io |
|---|---|---|
| 价格调整 | 不完整 | 完整（拆分 + 分红） |
| 历史成分股 | 无 | 有（通过 reference API） |
| 退市股票数据 | 无 | 有 |
| API 可靠性 | 无 SLA | 99.9% uptime |
| 数据延迟 | 15-20 分钟 | 实时（付费）/ 日终（免费） |

**免费层足够**：5 次/分钟的 API 调用限制对回测完全够用（我们不需要实时数据）。

注册地址：https://polygon.io/dashboard/signup

---

## 生存者偏差修正的实现方法

### 数据来源

使用 GitHub 上的公开数据集 `fja05680/sp500`，记录了标普 500 所有历史成分股变更。

### 修正原理

```
今天的标普 500 成分股列表
    │
    ├── 减去"后来才加入的股票"（比如 Tesla 2019年12月才加入）
    ├── 加回"后来被移除的股票"（比如 Lehman 2008年9月被移除）
    │
    └── = 目标日期的真实成分股列表
```

### 代码实现

```python
from src.survivorship_bias import SurvivorshipBiasCorrector

corrector = SurvivorshipBiasCorrector()
corrector.set_base_constituents("2024-01-01", current_sp500_tickers)

# 2008年9月的真实标普500（包含雷曼兄弟）
tickers_2008 = corrector.get_point_in_time_constituents("2008-09-01")
assert "LEH" in tickers_2008  # 雷曼兄弟还在

# 2015年的列表（不包含特斯拉）
tickers_2015 = corrector.get_point_in_time_constituents("2015-01-01")
assert "TSLA" not in tickers_2015  # 特斯拉还没加入
```

---

## 数据验证流程

所有数据在进入回测系统前，必须通过 6 项质量检查：

```python
from src.data_validator import DataValidator

validator = DataValidator()
report = validator.validate_ohlcv(df)

if report.data_quality_score < 0.9:
    print("数据质量不达标！")
    for error in report.errors:
        print(f"  错误: {error}")
    for warning in report.warnings:
        print(f"  警告: {warning}")
```

### 检查项目

1. **缺失交易日** — 和交易日历对比，找出缺失的日期
2. **价格异常** — 负价格、NaN 值
3. **OHLC 逻辑** — High 必须 >= Open/Close/Low
4. **未调整拆分** — 单日涨跌超 50% 大概率是数据问题
5. **价格冻结** — 连续 5+ 天收盘价相同（停牌或数据错误）
6. **零成交量** — 可能是停牌或数据缺失

### 质量评分

- **1.0** — 完美数据
- **0.9+** — 可以放心用
- **0.7-0.9** — 有些小问题，建议检查
- **<0.7** — 不建议用于回测
