# 协作流程指南 (Workflow Guide)

> 你和 Claude Code 之间的标准操作流程。
> 遵循这个流程可以最大化效率，同时保护你的数据安全和电脑性能。

---

## 你的角色 vs Claude 的角色

| 你负责 (Human) | Claude 负责 (AI) |
|---|---|
| 决定做什么功能 | 写代码 + 写测试 |
| 审批变更计划 | 解释每个改动的影响 |
| 提供 API key | 保护 API key 不泄露 |
| 最终 merge 决策 | 自检代码质量 |
| 说"提交"才提交 | 永远不自动 push |

---

## 标准工作流 (每次改动都走这个流程)

### Step 1: 你说需求
```
示例:
"我想让 alpha_filter 在拒绝因子时输出更详细的日志"
"帮我加一个新的 market regime: sideways（横盘）"
"这段代码是干什么的？"
```

### Step 2: Claude 出变更计划
Claude 会展示:
- **改什么**: 哪些文件、哪些函数
- **为什么**: 为什么需要这个改动
- **影响范围**: 会不会影响其他功能
- **验收标准**: 怎么确认改完是对的
- **风险**: 有什么可能出问题
- **性能影响**: 会不会影响你的游戏

### Step 3: 你审批
```
"可以，做吧"          → Claude 开始写码
"不要改 XX 部分"      → Claude 调整方案
"先解释一下 YY 是什么" → Claude 先解释再继续
"算了不做了"          → Claude 停止
```

### Step 4: Claude 写码 + 跑测试
- 写最小必要代码
- 运行 `pytest tests/ -v` 确认全部通过（当前 118 个测试）
- 告诉你改了什么、为什么

### Step 5: 你决定是否提交
```
"提交吧"    → Claude 做 git commit
"先不提交"   → 改动保留在本地，不 commit
"push 到 GitHub" → Claude push（会再确认一次）
```

---

## 安全红线 (Claude 绝对不会做的事)

1. 删除 `rdagent-quant-opt/` 目录之外的任何文件
2. 修改 Windows 系统设置、注册表
3. 把 API key 或密码写进代码
4. 没经过你同意就 `git push`
5. 安装系统级软件
6. 启动常驻后台进程
7. 访问你的个人文件（游戏、文档、照片等）

---

## 如何保护游戏性能

这个项目对你日常使用电脑的影响：

| 组件 | 影响 | 说明 |
|---|---|---|
| 代码本身 | 零影响 | 只是文本文件，不占资源 |
| pytest 测试 | 极低 | 3 秒跑完，纯 CPU 计算 |
| Docker (跑回测时) | 中等 | 限制 CPU=2核、内存=4GB |
| WSL | 低 | 只在你主动启动时运行 |
| venv | 零 | 只是一个文件夹 |

**关键**: Docker 和 WSL 只在你手动启动时才运行。
关掉终端 = 关掉所有项目进程。游戏时不需要担心。

---

## 隐私保护清单

- [x] `.gitignore` 已排除 `.env`（API key 文件）
- [x] `.gitignore` 已排除 `logs/`（运行日志）
- [x] `.gitignore` 已排除 `venv/`（环境文件）
- [x] `.gitignore` 已排除 `data/`（本地数据缓存）
- [x] 代码中不包含任何个人信息
- [x] 提交前 Claude 会检查是否有敏感信息
- [ ] (推荐) 你可以安装 `git-secrets` 来自动扫描

---

## 常用命令速查

```bash
# === 进入项目环境 ===
wsl                                              # 进入 Linux 环境
cd $PROJECT_ROOT
source venv/bin/activate                         # 激活 Python 环境

# === 开发相关 ===
python -m pytest tests/ -v                       # 跑测试（3秒，118个测试）
python -m pytest tests/test_pipeline.py -v       # 只跑某个测试文件

# === 跑回测（需要 Docker + API key）===
export DEEPSEEK_API_KEY="sk-你的key"
bash scripts/run_backtest.sh budget 30           # 预算模式，30轮
bash scripts/run_backtest.sh optimized 10        # 优化模式，10轮（试水）
bash scripts/run_backtest.sh premium 30          # 最强模式，30轮

# === Git 相关 ===
git status                                       # 看有什么改动
git log --oneline -5                             # 看最近5次提交
git diff                                         # 看具体改了什么

# === 退出 ===
deactivate                                       # 退出 Python 环境
exit                                             # 退出 WSL
```

---

## 如何跑通整个系统 (Step-by-Step)

### 阶段一: 环境准备（一次性，约30分钟）

```bash
# 1. 安装 WSL（如果还没有的话）
#    在 Windows PowerShell (管理员) 中运行:
wsl --install

# 2. 安装 Docker Desktop
#    下载: https://www.docker.com/products/docker-desktop/
#    安装后，在 Settings > Resources 中:
#    - CPUs: 设为 2（保留其他给游戏）
#    - Memory: 设为 4 GB（你有 32GB，这很安全）
#    - 勾选 "Use the WSL 2 based engine"

# 3. 进入 WSL，准备 Python 环境
wsl
cd $PROJECT_ROOT
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. 获取 API Keys
#    DeepSeek（必须）: https://platform.deepseek.com/ — 新号送500万tokens
#    Polygon.io（推荐）: https://polygon.io/dashboard/signup — 免费层够用
#    Anthropic（可选）: https://console.anthropic.com/ — 只有premium模式需要

# 5. 配置环境变量
cp .env.example .env
#    编辑 .env 文件，填入你的 API key
```

### 阶段二: 验证安装（5分钟）

```bash
# 1. 跑测试，确认代码正常
source venv/bin/activate
python -m pytest tests/ -v
# 期望结果: 118 passed

# 2. 确认 Docker 在运行
docker info
# 如果报错，去 Docker Desktop 点启动
```

### 阶段三: 首次回测（试水，约1小时）

```bash
# 用最便宜的配置，只跑5轮试试
export DEEPSEEK_API_KEY="sk-你的key"
bash scripts/run_backtest.sh budget 5

# 预计花费: < $0.1
# 这一步是为了确认整个链路能跑通
```

### 阶段四: 正式回测（约5-10小时）

```bash
# 确认试水成功后，跑完整30轮
bash scripts/run_backtest.sh optimized 30
# optimized 模式: 构思用 Reasoner，代码用 Chat
# 预计花费: $1.5 - $3.0

# 如果想用最强模型（Opus 4.6 做因子构思）:
bash scripts/run_backtest.sh premium 30
# 预计花费: $15 - $30（需要 Anthropic API key）

# 结果保存在 logs/ 目录下
```

### 阶段五: 查看结果 & 推到 GitHub

```bash
# 对比不同配置的结果
python3 scripts/compare_runs.py

# 推到 GitHub（让 Claude 帮你做）
# 告诉 Claude: "帮我创建 GitHub 仓库并 push"
```

---

## 目标市场说明

本系统主要面向 **美股 S&P 500（标普500）**：
- 数据来源: Polygon.io（主）+ Yahoo Finance（备用）
- 成分股: 500 只美国大盘股
- 时间跨度: 2010-2024（默认配置）
- 数据频率: 日线（每天一条数据）
- 生存者偏差修正: 包含已退市股票（如 2008 年的雷曼兄弟）

也兼容 **CSI 300（沪深300）**（通过 Qlib + Yahoo Finance）。

---

## 模型选择逻辑

系统会根据每个任务的特征自动选择最合适的模型：

| 任务 | 特征 | 选择逻辑 | 结果 |
|---|---|---|---|
| **因子构思** (Synthesis) | 需要创造力，调用少 | 高创造力 + 低频 → frontier | Opus 4.6 |
| **结果分析** (Analysis) | 需要因果推理，调用少 | 高推理 + 低频 → strong | DeepSeek Reasoner |
| **代码生成** (Implementation) | 结构化任务，调用多，有重试 | 高频 + 可重试 → efficient | DeepSeek Chat |
| **模板组装** (Specification) | 简单任务 | 低需求 → efficient | DeepSeek Chat |

**边际收益原则**: 代码生成占 60-70% 的调用量，但有重试机制（错了会自动重试最多 10 次），
所以用便宜模型就够了——换贵模型提升很小但成本翻几十倍。
而因子构思每轮只调 1-2 次，且没有重试，输出质量决定整个迭代方向，所以值得用最强模型。

三种预算模式对比：

| 模式 | 构思模型 | 分析模型 | 代码模型 | 30轮预估成本 |
|---|---|---|---|---|
| `budget` | DeepSeek Chat | DeepSeek Chat | DeepSeek Chat | $0.5 - $1.5 |
| `optimized` | Reasoner | Reasoner | Chat | $1.5 - $3.0 |
| `premium` | Opus 4.6 | Reasoner | Chat | $15 - $30 |

---

## FAQ

**Q: 跑回测的时候可以打游戏吗？**
A: 可以。Docker 已限制为 2核 + 4GB 内存，你的电脑有 32GB 内存和多核 CPU，影响很小。如果感觉卡，可以暂停 Docker。

**Q: API key 会不会泄露？**
A: 不会。`.env` 文件在 `.gitignore` 里，永远不会被 git 提交。Claude 也被禁止把 key 写进代码。

**Q: 跑一次要花多少钱？**
A: Budget 模式 30 轮约 $0.5-1.5，Optimized 模式约 $1.5-3.0。DeepSeek 新号送的额度够跑很多次。

**Q: 万一 Claude 改错了代码怎么办？**
A: 用 `git diff` 看改了什么，`git checkout -- 文件名` 撤销某个文件的改动，或 `git stash` 暂存所有改动。

**Q: 需要一直开着 WSL 吗？**
A: 不需要。只在开发或跑回测时打开。关掉终端窗口就自动停止了。

**Q: 这个系统做哪个市场？**
A: 主要做美股标普500（S&P 500），也兼容中国A股沪深300（CSI 300）。

**Q: 数据准确吗？**
A: 系统有 6 项数据质量检查（负价格、缺失日期、未调整拆分等），质量评分低于 0.9 会自动警告。Polygon.io 提供专业级交易数据。

**Q: 什么是生存者偏差？**
A: 如果你只用"现在还活着"的股票做回测，会跳过倒闭/退市的公司，导致结果虚高。我们的系统用历史成分股记录还原了真实的股票池，包含已退市的公司。
