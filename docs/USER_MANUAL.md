# 使用手册 — 从零开始玩转量化因子研发系统

> 写给完全没有编程经验的人。
> 如果你能看懂微信，你就能看懂这份手册。

> **注意**: 本文档中所有 `$PROJECT_ROOT` 指的是你把项目放在哪里的路径。
> 比如你把项目 clone 到了 `/home/user/rdagent-quant-opt`，那就把 `$PROJECT_ROOT` 替换成它。
> 在 WSL 中，Windows 路径通常是 `/mnt/c/rdagent-quant-opt`。

---

## 目录

1. [这个系统是干什么的？](#1-这个系统是干什么的)
2. [它做哪个市场？](#2-它做哪个市场)
3. [从零搭建环境](#3-从零搭建环境)
4. [第一次运行](#4-第一次运行)
5. [如何自动化生产因子](#5-如何自动化生产因子)
6. [如何管理和查询因子](#6-如何管理和查询因子)
7. [如何测试自己写的因子](#7-如何测试自己写的因子)
8. [如何更换模型](#8-如何更换模型)
9. [数据从哪来？准不准？](#9-数据从哪来准不准)
10. [花多少钱？](#10-花多少钱)
11. [常见问题](#11-常见问题)

---

## 1. 这个系统是干什么的？

**一句话**：让 AI 自动帮你想出赚钱的股票交易策略，然后用历史数据验证它到底能不能赚钱。

**详细说**：

1. AI 提出一个假设（比如"过去 20 天涨幅最大的股票，未来还会继续涨"）
2. AI 把这个假设写成可执行的代码
3. 系统用过去 10 多年的真实股票数据去模拟交易
4. 另外两个 AI（一个看多、一个看空）辩论这个策略到底好不好
5. 系统决定是继续优化这个策略，还是换一个方向
6. 重复 30 轮，最终找到表现最好的策略

**你不需要会写代码**。系统全自动运行，你只需要：
- 按照本手册搭好环境
- 输入一条命令启动
- 等它跑完看结果

---

## 2. 它做哪个市场？

### 主要市场：美股标普 500（S&P 500）

| 项目 | 说明 |
|---|---|
| 市场 | 美国股票市场 |
| 股票范围 | 标普 500 指数的 500 只大盘股（苹果、微软、谷歌等） |
| 数据时间 | 2010 年至 2024 年（默认） |
| 数据频率 | 每天一条数据（日线） |
| 交易模拟 | 包含手续费和滑点（接近真实交易成本） |

### 为什么选标普 500？

- **数据公开免费**：不需要开户、不需要付费订阅
- **流动性好**：大盘股不容易出现"买不到/卖不掉"的问题
- **研究最充分**：学术论文最常用的市场，方便对比
- **数据质量高**：Polygon.io 提供交易级别的精确数据

### 也支持中国 A 股（CSI 300）

通过 Qlib + Yahoo Finance 数据，可以跑沪深 300 的回测。但因为 Yahoo Finance 的 A 股数据质量不如美股，推荐优先用标普 500。

---

## 3. 从零搭建环境

### 你需要准备的东西

| 必须 | 说明 | 在哪拿 |
|---|---|---|
| Windows 电脑 | 你已经有了 | — |
| WSL | Windows 上运行 Linux 的工具 | 见下面 Step 1 |
| Docker Desktop | 运行回测代码的安全沙箱 | 见下面 Step 2 |
| DeepSeek API Key | AI 模型的"钥匙" | 见下面 Step 4 |

| 推荐（非必须） | 说明 | 在哪拿 |
|---|---|---|
| Polygon.io API Key | 更准确的股票数据 | https://polygon.io/dashboard/signup |
| Anthropic API Key | 最强 AI 模型（premium 模式） | https://console.anthropic.com/ |

### Step 1: 安装 WSL

WSL 是什么？它让你在 Windows 上运行 Linux 命令。我们的系统需要 Linux 环境。

1. 右键点击 Windows 开始菜单 → **终端（管理员）**
2. 输入：`wsl --install`
3. 等它装完，**重启电脑**
4. 重启后会自动弹出 Ubuntu 窗口，设置一个用户名和密码（这是 Linux 的密码，不影响 Windows）

### Step 2: 安装 Docker Desktop

Docker 是什么？它是一个"安全容器"——AI 生成的代码在里面运行，不会影响你的电脑。

1. 打开浏览器，搜索 "Docker Desktop 下载"
2. 安装后打开 Docker Desktop
3. 进入 **Settings → Resources**：
   - **CPUs**: 设为 **2**（保留其他给游戏）
   - **Memory**: 设为 **4 GB**（你有 32GB，很安全）
   - 勾选 **"Use the WSL 2 based engine"**
4. 点 **Apply & Restart**

### Step 3: 安装项目

打开你的终端（或 WSL 窗口），输入以下命令：

```bash
# 进入 WSL
wsl

# 进入项目目录
cd $PROJECT_ROOT

# 创建 Python 虚拟环境（第一次才需要）
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装所有依赖
pip install -r requirements.txt
```

### Step 4: 获取 API Key

**DeepSeek（必须，免费额度充足）**：
1. 打开 https://platform.deepseek.com/
2. 注册账号
3. 进入 API Keys 页面，创建一个 key
4. 复制 key（类似 `sk-xxxxxxxxxxxxxxxx`）
5. 新号送 500 万 tokens，够跑好几次完整回测

**Polygon.io（推荐，免费）**：
1. 打开 https://polygon.io/dashboard/signup
2. 注册账号
3. 进入 Dashboard，复制你的 API Key
4. 免费层：每分钟 5 次请求，对回测完全够用

### Step 5: 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 用文本编辑器打开 .env 文件
nano .env
```

找到这两行，把 `your-key-here` 替换成你的真实 key：
```
DEEPSEEK_API_KEY=sk-你从DeepSeek复制的key
POLYGON_API_KEY=你从Polygon复制的key
```

保存退出：按 `Ctrl+X`，然后按 `Y`，再按 `Enter`。

### Step 6: 验证安装

```bash
# 跑一下测试，确认所有东西都装好了
python -m pytest tests/ -v
```

你应该看到：**118 passed**。如果全绿，恭喜，环境搭好了。

---

## 4. 第一次运行

### 试水（5 轮，花费 < $0.1）

```bash
# 确保在项目目录里，且虚拟环境已激活
cd $PROJECT_ROOT
source venv/bin/activate

# 确保 Docker Desktop 已启动

# 跑 5 轮试试（最便宜的 budget 模式）
bash scripts/run_backtest.sh budget 5
```

你会看到：
- 系统自动下载股票数据（第一次需要几分钟）
- AI 开始生成因子假设
- 代码被自动生成和调试
- 回测结果出来
- 成本报告显示总共花了多少钱

### 看结果

```bash
# 查看最近一次运行的结果
ls logs/

# 对比不同运行的结果
python3 scripts/compare_runs.py
```

---

## 5. 如何自动化生产因子

### 完整 30 轮运行

```bash
# 推荐模式：Reasoner 想思路 + Chat 写代码（$1.5-3.0）
bash scripts/run_backtest.sh optimized 30

# 最便宜：全用 Chat（$0.5-1.5）
bash scripts/run_backtest.sh budget 30

# 最强：Opus 想思路 + Reasoner 分析 + Chat 写代码（$15-30）
bash scripts/run_backtest.sh premium 30
```

### 系统每轮自动做什么？

```
第 1 轮:
  → 检测市场状态（牛市？熊市？高波动？）
  → AI 根据市场状态想出一个因子假设
  → 检查这个因子是不是和之前的重复（防止"因子拥挤"）
  → AI 写代码实现这个因子
  → 用历史数据回测这个因子
  → 两个 AI（看多 vs 看空）辩论这个因子好不好
  → 决定下一轮是继续优化还是换方向

第 2 轮:
  → 根据上一轮的辩论结果，决定策略方向
  → 重复上述流程...

...

第 30 轮:
  → 输出最终报告：最好的因子、成本汇总、辩论记录
```

### 运行期间可以干什么？

- **打游戏**：完全没问题。Docker 被限制为 2 核 + 4GB 内存
- **关掉终端**：回测会停止。如果想后台运行，用 `nohup bash scripts/run_backtest.sh optimized 30 &`
- **随时查看进度**：打开另一个终端窗口，看 `logs/` 下的日志文件

---

## 6. 如何管理和查询因子

### 查看已生成的因子

每次运行的因子保存在 `logs/run_日期_模式/` 目录下：

```bash
# 列出所有运行记录
ls logs/

# 查看某次运行的结果
cat logs/run_20260319_120530_optimized/results.json

# 查看成本报告
cat logs/run_20260319_120530_optimized/cost_log.json
```

### 对比不同运行

```bash
python3 scripts/compare_runs.py
```

这会输出一个对比表，显示每次运行的：
- 使用的模式（budget/optimized/premium）
- 总花费
- API 调用次数
- 每个阶段的成本分布

### 在 Python 中查询因子

```python
from src.pipeline import OptimizedPipeline

pipeline = OptimizedPipeline("configs/default.yaml")

# 查看因子库中已有的因子
for name, code in pipeline.accepted_factors:
    print(f"因子: {name}")
    print(f"代码: {code[:100]}...")
    print()
```

---

## 7. 如何测试自己写的因子

### 方法 1：用抗衰减过滤器检查

你可以把自己写的因子代码丢给系统，它会帮你检查：
- 是不是和已有的因子太像（可能没用）
- 是不是太复杂（可能过拟合）

```python
from src.alpha_filter import AlphaDecayFilter

# 创建过滤器
checker = AlphaDecayFilter(
    similarity_threshold=0.85,
    max_complexity_depth=5,
)

# 注册一些已有的因子（用来对比）
checker.add_existing_factor("momentum_20d", "df['close'].pct_change(20)")
checker.add_existing_factor("volatility", "df['close'].rolling(20).std()")

# 检查你自己的因子
my_factor_code = """
factor = df['close'].pct_change(5) / df['close'].rolling(10).std()
"""

result = checker.evaluate(my_factor_code, check_alignment=False)

if result.passed:
    print("通过！这个因子足够独特且不会过拟合。")
    print(f"  相似度: {result.similarity_score:.2f}")
    print(f"  复杂度: {result.complexity_depth}")
else:
    print("未通过！原因：")
    for reason in result.rejection_reasons:
        print(f"  - {reason}")
```

### 方法 2：用市场状态检测器了解当前市场

```python
from src.market_regime import MarketRegimeDetector

detector = MarketRegimeDetector()

# 用规则检测（免费，不花钱）
regime = detector.detect_from_indicators(
    returns_20d=0.05,       # 过去20天收益率 5%
    volatility_20d=0.12,    # 20日波动率 12%
    ma_cross=0.03,          # 均线交叉幅度
    vol_percentile=0.3,     # 波动率在历史中的百分位
)

print(f"市场状态: {regime.label}")
print(f"描述: {regime.description}")
print(f"置信度: {regime.confidence:.2f}")
# 输出示例: "市场状态: bull_low_trend, 置信度: 0.85"
```

### 方法 3：用辩论系统评估策略方向

```python
from src.debate_agents import DebateAnalyzer

analyzer = DebateAnalyzer()

# 注意：这需要 API key 和网络连接，会花少量钱（约 $0.007）
result = analyzer.debate(
    hypothesis="基于5日动量和10日波动率的组合因子",
    code="df['close'].pct_change(5) / df['close'].rolling(10).std()",
    metrics={
        "IC": 0.025,           # 信息系数
        "ICIR": 0.12,          # 信息系数比率
        "annual_return": 0.06, # 年化收益
        "max_drawdown": -0.15, # 最大回撤
    },
    iteration=3,
)

print(f"裁决: {result.verdict.value}")  # CONTINUE / PIVOT / NEUTRAL
print(f"置信度: {result.confidence:.2f}")
print(f"看多观点: {result.bull_argument[:100]}...")
print(f"看空观点: {result.bear_argument[:100]}...")
print(f"建议: {result.recommended_action}")
```

---

## 8. 如何更换模型

### 方式 1：换预设模式（最简单）

运行时直接换模式名字就行：

```bash
# 全用便宜模型
bash scripts/run_backtest.sh budget 30

# 推理用强模型，代码用便宜模型（推荐）
bash scripts/run_backtest.sh optimized 30

# 最强模型做构思
bash scripts/run_backtest.sh premium 30
```

### 方式 2：在配置文件里改（自定义）

编辑 `configs/default.yaml`：

```yaml
# 改模型注册表中的模型 ID
models:
  frontier:
    model_id: "anthropic/claude-opus-4-6"   # ← 换成你想用的模型
  strong:
    model_id: "deepseek/deepseek-reasoner"  # ← 换成你想用的模型
  efficient:
    model_id: "deepseek/deepseek-chat"      # ← 换成你想用的模型
```

### 方式 3：通过环境变量覆盖（临时）

```bash
# 只这一次用不同的模型
export SYNTHESIS_MODEL=anthropic/claude-opus-4-6
export ANALYSIS_MODEL=deepseek/deepseek-reasoner
export IMPLEMENTATION_MODEL=deepseek/deepseek-chat
bash scripts/run_backtest.sh optimized 30
```

### 当前支持的模型

| 模型 | 提供商 | 用途 | 价格级别 |
|---|---|---|---|
| claude-opus-4-6 | Anthropic | 最强创造力 | 贵（$15/M input） |
| claude-sonnet-4-6 | Anthropic | 高性价比 | 中等（$3/M input） |
| deepseek-reasoner | DeepSeek | 强推理能力 | 便宜（$0.55/M input） |
| deepseek-chat | DeepSeek | 高速度低成本 | 最便宜（$0.27/M input） |

### 系统如何自动选模型？

系统根据每个任务的特点自动选：
- **因子构思**：需要创造力，调用少 → 用最强模型
- **代码生成**：调用多（每轮 10-15 次），错了会自动重试 → 用便宜模型
- **结果分析**：需要深度推理 → 用推理型模型

这叫**边际收益原则**：在提升最大的地方花钱，在提升最小的地方省钱。

---

## 9. 数据从哪来？准不准？

### 数据来源

| 数据源 | 角色 | 质量 | 费用 |
|---|---|---|---|
| **Polygon.io** | 主数据源 | 专业级（交易所直接数据） | 免费层够用 |
| **Yahoo Finance** | 备用数据源 | 尚可（部分拆分调整有误差） | 完全免费 |

系统自动优先用 Polygon，如果 Polygon 出问题就回退到 Yahoo。你不需要关心这个过程。

### 数据精度

系统在使用数据前会做 **6 项质量检查**：

| 检查项 | 检查什么 | 为什么重要 |
|---|---|---|
| 缺失日期 | 交易日是否有缺失 | 缺数据会导致因子计算错误 |
| 价格异常 | 负价格、空值 | 现实中不可能，说明数据有错 |
| OHLC 一致性 | 最高价是否 >= 最低价 | 逻辑上必须成立 |
| 拆分调整 | 单日涨跌 > 50% | 大概率是股票拆分没调整 |
| 停牌检测 | 连续 5+ 天同价 | 可能停牌或数据冻结 |
| 成交量检查 | 零成交量 | 可能停牌或数据缺失 |

每份数据会得到一个 **0.0-1.0 的质量评分**。低于 0.9 系统会警告。

### 生存者偏差修正

**问题**：如果只用"现在还活着"的股票做回测，会跳过倒闭/退市的公司，导致结果虚高。

**解决**：系统用标普 500 的历史成分股变更记录，还原每个历史日期的真实股票池。比如 2008 年 9 月的回测会包含当时还没倒闭的雷曼兄弟。

### 数据更新

- Polygon 免费层提供日终数据（每天收盘后更新）
- 运行回测时系统自动下载最新数据
- 历史数据只需下载一次，之后缓存在本地

---

## 10. 花多少钱？

### API 费用

| 模式 | 30 轮花费 | 适合谁 |
|---|---|---|
| **budget** | $0.5 - $1.5 | 试水、学习 |
| **optimized** | $1.5 - $3.0 | 正式使用（推荐） |
| **premium** | $15 - $30 | 追求最高质量 |

### 免费额度

- DeepSeek 新号送 **500 万 tokens**，约等于 $1.40
- 够跑 **1-2 次完整 budget 模式**的 30 轮回测
- Polygon.io 免费层**永久免费**，不限调用次数（只限速度）

### 省钱技巧

1. 先用 `budget 5` 试水（花费 < $0.1）
2. 确认没问题再跑 `optimized 30`
3. 代码生成永远用便宜模型（系统自动的，不用你管）
4. DeepSeek 的缓存机制会自动帮你省钱（重复的 prompt 打 9 折）

---

## 11. 常见问题

### 安装类

**Q: `wsl --install` 报错？**
A: 确保用管理员权限运行。右键开始菜单 → "终端（管理员）"。

**Q: Docker Desktop 打不开？**
A: 确保 BIOS 里开启了虚拟化（Virtualization）。大部分电脑默认开启。

**Q: `pip install` 报错？**
A: 确保在 WSL 里运行，不是在 Windows CMD 里。先输入 `wsl` 进入 Linux 环境。

**Q: 测试没有跑出 118 passed？**
A: 可能是依赖没装全。运行 `pip install -r requirements.txt` 重新安装。

### 运行类

**Q: 回测跑到一半报错了？**
A: 查看 `logs/` 目录下的日志文件。最常见原因是 API key 过期或额度用完。

**Q: 跑回测时电脑很卡？**
A: 打开 Docker Desktop → Settings → Resources，把 CPU 从 2 降到 1。

**Q: 回测跑了很久没反应？**
A: 正常。30 轮回测需要 5-10 小时。可以看日志文件确认是否在运行。

**Q: API key 泄露了怎么办？**
A: 立即去对应平台（DeepSeek/Polygon/Anthropic）的设置页面删除旧 key，创建新 key。你的 `.env` 文件不会被 git 提交，所以不会传到网上。

### 结果类

**Q: 因子的 IC 是什么？**
A: IC (Information Coefficient) 是衡量因子预测能力的指标。0 = 没有预测力，0.05+ = 不错，0.1+ = 很强。

**Q: 什么样的结果算好？**
A: IC > 0.03、ICIR > 0.1、年化收益 > 5%、最大回撤 < 20% 是一个不错的起点。

**Q: 为什么不同运行的结果不一样？**
A: AI 模型有随机性。同样的设置跑两次，得到的因子假设可能不同。这是正常的。多跑几次取最好的。

**Q: 结果能直接用来炒股吗？**
A: **不能直接用**。回测结果和真实交易有差距（数据延迟、滑点、流动性等）。这个系统是研究工具，不是交易系统。任何投资决策请自己负责。

### 费用类

**Q: 会不会不小心花很多钱？**
A: 不会。系统有每日预算限制（默认 $5/天）。超过预算会自动停止。你可以在 `configs/default.yaml` 里调整 `daily_budget_usd`。

**Q: DeepSeek 免费额度用完了怎么办？**
A: 充值。DeepSeek 的价格非常便宜，$10 够跑很多次。

---

## 快速参考卡片

```
┌─────────────────────────────────────────────┐
│  快速启动                                    │
│                                             │
│  1. wsl                                     │
│  2. cd $PROJECT_ROOT                        │
│                                             │
│  3. source venv/bin/activate                │
│  4. bash scripts/run_backtest.sh budget 5   │
│                                             │
│  跑测试: python -m pytest tests/ -v         │
│  看结果: python3 scripts/compare_runs.py    │
│  看文档: docs/ 目录下                        │
└─────────────────────────────────────────────┘
```
