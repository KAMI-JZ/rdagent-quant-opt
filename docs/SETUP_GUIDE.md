# WSL2 + Qlib + RD-Agent 环境搭建指南

> **目标**：让 rdagent-quant-opt 项目跑通真实回测，产出真实 IC/ICIR/Sharpe 数据
> **预计总耗时**：1-1.5 小时（大部分时间在等数据下载）
> **前提**：Windows 11 Home，BIOS 虚拟化已开启

---

## 阶段 0：修复 WSL2（约 15 分钟）

### 问题
WSL2 报错 `HCS_E_SERVICE_NOT_AVAILABLE`，Hyper-V 计算服务未运行。

### 步骤

**0.1 以管理员身份打开 PowerShell**
- 右键「开始」菜单 → 选「终端(管理员)」或「Windows PowerShell(管理员)」

**0.2 启用必要的 Windows 功能（逐行复制粘贴执行）**
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

**0.3 重启电脑**
```powershell
shutdown /r /t 0
```

**0.4 重启后验证 WSL**
打开普通终端（不需要管理员），执行：
```powershell
wsl -d Ubuntu -- echo "WSL works!"
```
✅ 看到 `WSL works!` → 进入阶段 1
❌ 仍然失败 → 执行 0.5

**0.5 如果仍然失败（备选方案）**
管理员 PowerShell 执行：
```powershell
bcdedit /set hypervisorlaunchtype auto
shutdown /r /t 0
```
重启后再次验证 `wsl -d Ubuntu -- echo "WSL works!"`

### 参考链接
- WSL 安装官方文档：https://learn.microsoft.com/zh-cn/windows/wsl/install
- WSL 疑难解答：https://learn.microsoft.com/zh-cn/windows/wsl/troubleshooting
- Windows Home 启用 Hyper-V：https://github.com/AveYo/Virtual-Desktop-Bar

---

## 阶段 1：WSL 内安装 Qlib（约 30 分钟）

以下所有命令在 **WSL Ubuntu 终端** 中执行。

**1.1 安装基础依赖**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-venv python3-pip git build-essential
```

**1.2 创建项目虚拟环境**
```bash
cd ~
python3 -m venv rdagent-env
source ~/rdagent-env/bin/activate
pip install --upgrade pip setuptools wheel
```

**1.3 安装 Qlib（微软量化回测框架）**
```bash
pip install pyqlib
```

**1.4 下载美股数据（约 2-3 GB，需要几分钟）**
```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

**1.5 验证 Qlib 安装**
```bash
python3 -c "
import qlib
qlib.init(provider_uri='~/.qlib/qlib_data/us_data', region_type='us')
from qlib.data import D
df = D.features(['AAPL'], ['\$close'], start_time='2020-01-01', end_time='2020-01-31')
print(df.head())
print('Qlib works! Rows:', len(df))
"
```
✅ 看到 AAPL 收盘价数据 → 进入阶段 2

### 参考链接
- Qlib 安装文档：https://qlib.readthedocs.io/en/latest/start/installation.html
- Qlib GitHub：https://github.com/microsoft/qlib
- Qlib 数据准备：https://qlib.readthedocs.io/en/latest/start/getdata.html

---

## 阶段 2：安装 RD-Agent（约 20 分钟）

**2.1 安装 RD-Agent（在同一个 venv 中）**
```bash
source ~/rdagent-env/bin/activate

# 从 GitHub 安装最新版（推荐）
pip install git+https://github.com/microsoft/RD-Agent.git
```

**2.2 确保 Docker 可用**

首先确认 Windows 侧 Docker Desktop 已启动，然后：
- 打开 Docker Desktop → Settings → Resources → WSL Integration
- 勾选 ✅ Enable integration with my default WSL distro
- 勾选 ✅ Ubuntu
- 点 Apply & Restart

回到 WSL 验证：
```bash
docker run --rm hello-world
```
✅ 看到 `Hello from Docker!` → Docker 可用

### 参考链接
- Docker Desktop WSL 集成：https://docs.docker.com/desktop/features/wsl/

**2.3 配置环境变量**
```bash
cat >> ~/.bashrc << 'EOF'

# ===== RD-Agent Quant Optimizer 配置 =====
export DEEPSEEK_API_KEY="sk-你的DeepSeek-API-Key"
export CHAT_MODEL="deepseek/deepseek-chat"
export EMBEDDING_MODEL="BAAI/bge-m3"

# Docker 资源限制（保护游戏性能）
export DOCKER_CPUS=2
export DOCKER_MEMORY=4g

# Polygon.io（可选，没有也能用 Yahoo Finance）
# export POLYGON_API_KEY="你的Polygon-Key"
EOF

source ~/.bashrc
```

> ⚠️ 把 `sk-你的DeepSeek-API-Key` 替换为你的真实 API Key
> DeepSeek API Key 申请：https://platform.deepseek.com/api_keys （免费 500 万 token）

**2.4 验证 RD-Agent**
```bash
python3 -c "
from rdagent.core.scenario import Scenario
print('RD-Agent imported successfully')
"
```
✅ 看到 `RD-Agent imported successfully` → 进入阶段 3

### 参考链接
- RD-Agent GitHub：https://github.com/microsoft/RD-Agent
- RD-Agent 文档：https://microsoft.github.io/RD-Agent/
- RD-Agent 量化场景：https://rdagent.azurewebsites.net/factor_extraction_and_implementation
- DeepSeek API：https://platform.deepseek.com/api_keys

---

## 阶段 3：安装项目依赖（约 5 分钟）

**3.1 复制项目到 WSL 原生文件系统（推荐，IO 快 10 倍）**
```bash
source ~/rdagent-env/bin/activate
cp -r /mnt/c/rdagent-quant-opt ~/rdagent-quant-opt
cd ~/rdagent-quant-opt
pip install -r requirements.txt
```

**3.2 验证全部 12 个模块导入正常**
```bash
cd ~/rdagent-quant-opt
python3 -c "
from src import (
    MultiModelRouter, AlphaDecayFilter, DebateAnalyzer,
    MarketRegimeDetector, OptimizedPipeline, ExperienceMemory,
    ParameterOptimizer, TrajectoryEvolver, ReportGenerator,
    DataValidator, HybridProvider, SurvivorshipBiasCorrector
)
print('All 12 modules imported successfully!')
"
```
✅ 看到 `All 12 modules imported successfully!` → 进入阶段 4

**3.3 跑一次测试确认基础功能正常**
```bash
cd ~/rdagent-quant-opt
python -m pytest tests/ -v --tb=short
```
✅ 189 tests passed → 一切正常

---

## 阶段 4：回到 Claude Code

打开 Claude Code，对我说：

> **"环境准备好了"**

我会立即开始：
1. 写 `src/qlib_backtester.py` — 真实 Qlib 回测引擎
2. 替换 `pipeline.py` 中的 `_run_backtest()` 假数据
3. 接入 `param_optimizer.py` 的真实目标函数
4. 写集成测试 `tests/test_integration.py`
5. 跑通 3-5 轮真实迭代，产出真实 IC/ICIR/Sharpe 数据

---

## 故障排除

### WSL 相关
| 问题 | 解决方案 |
|------|----------|
| `HCS_E_SERVICE_NOT_AVAILABLE` | 阶段 0.2 + 0.3（启用功能+重启） |
| WSL 启动慢 | 正常，第一次启动需要 30-60 秒 |
| `wsl --list` 没有 Ubuntu | `wsl --install -d Ubuntu` |

### Docker 相关
| 问题 | 解决方案 |
|------|----------|
| `docker: command not found` (WSL内) | Docker Desktop → Settings → WSL Integration → 启用 Ubuntu |
| Docker Desktop 打不开 | 需要先修好 WSL2（阶段 0） |
| `permission denied` | `sudo usermod -aG docker $USER` 然后重新打开终端 |

### Qlib 相关
| 问题 | 解决方案 |
|------|----------|
| `pip install pyqlib` 失败 | `pip install cython numpy` 先装依赖，再装 qlib |
| 数据下载失败 | 检查网络，或用代理：`export https_proxy=http://...` |
| `import qlib` 报错 | 确认在 venv 里：`which python3` 应该指向 `~/rdagent-env/bin/python3` |

### RD-Agent 相关
| 问题 | 解决方案 |
|------|----------|
| 安装超时 | `pip install --default-timeout=100 git+https://...` |
| 依赖冲突 | `pip install --no-deps rdagent` 然后手动装缺的包 |
| import 报错 | 检查版本：`pip show rdagent` |
