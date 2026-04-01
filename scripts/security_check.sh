#!/bin/bash
# ============================================================
# 隐私安全检查脚本 (Privacy Security Check)
# 在 push 到 GitHub 之前运行此脚本
# Usage: bash scripts/security_check.sh
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
WARN=0

echo "======================================="
echo "  Privacy Security Check"
echo "  隐私安全检查"
echo "======================================="
echo ""

# ── 1. 检查 .env 文件是否被 git 追踪 ──
echo -n "[1/7] .env file not tracked by git... "
if git ls-files --error-unmatch .env 2>/dev/null; then
    echo -e "${RED}FAIL${NC} — .env is tracked! Run: git rm --cached .env"
    FAIL=$((FAIL+1))
else
    echo -e "${GREEN}PASS${NC}"
    PASS=$((PASS+1))
fi

# ── 2. 检查 .gitignore 是否包含关键排除项 ──
echo -n "[2/7] .gitignore has required exclusions... "
MISSING=""
for pattern in ".env" "logs/" "data/" "venv/" "__pycache__/"; do
    if ! grep -q "^${pattern}$\|^${pattern}[[:space:]]" .gitignore 2>/dev/null; then
        MISSING="${MISSING} ${pattern}"
    fi
done
if [ -z "$MISSING" ]; then
    echo -e "${GREEN}PASS${NC}"
    PASS=$((PASS+1))
else
    echo -e "${RED}FAIL${NC} — Missing:${MISSING}"
    FAIL=$((FAIL+1))
fi

# ── 3. 检查所有待提交文件中是否有真实 API key ──
echo -n "[3/7] No real API keys in tracked files... "
# 检查 git 追踪的所有文件
KEY_FOUND=""
for f in $(git ls-files); do
    if [ -f "$f" ]; then
        # 匹配常见 API key 格式（排除占位符）
        if grep -qE "sk-[a-zA-Z0-9]{20,}" "$f" 2>/dev/null; then
            # 排除明显的占位符
            if ! grep -qE "sk-your|sk-xxx|sk-example|sk-placeholder|sk-..." "$f" 2>/dev/null; then
                if grep -qE "sk-[a-f0-9]{32}" "$f" 2>/dev/null; then
                    KEY_FOUND="${KEY_FOUND} ${f}"
                fi
            fi
        fi
    fi
done
if [ -z "$KEY_FOUND" ]; then
    echo -e "${GREEN}PASS${NC}"
    PASS=$((PASS+1))
else
    echo -e "${RED}FAIL${NC} — Found in:${KEY_FOUND}"
    FAIL=$((FAIL+1))
fi

# ── 4. 检查是否有硬编码的个人路径 ──
echo -n "[4/7] No hardcoded personal paths... "
PERSONAL_PATHS=""
for f in $(git ls-files); do
    if [ -f "$f" ]; then
        if grep -qiE "/Users/[a-zA-Z0-9_-]+/|C:\\\\Users\\\\[a-zA-Z0-9_-]+\\\\" "$f" 2>/dev/null; then
            PERSONAL_PATHS="${PERSONAL_PATHS} ${f}"
        fi
    fi
done
if [ -z "$PERSONAL_PATHS" ]; then
    echo -e "${GREEN}PASS${NC}"
    PASS=$((PASS+1))
else
    echo -e "${RED}FAIL${NC} — Found in:${PERSONAL_PATHS}"
    FAIL=$((FAIL+1))
fi

# ── 5. 检查 .env.example 是否安全（只有占位符）──
echo -n "[5/7] .env.example has only placeholders... "
if [ -f ".env.example" ]; then
    if grep -qE "sk-[a-f0-9]{32}" .env.example 2>/dev/null; then
        echo -e "${RED}FAIL${NC} — Real key found in .env.example!"
        FAIL=$((FAIL+1))
    else
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS+1))
    fi
else
    echo -e "${YELLOW}WARN${NC} — .env.example not found"
    WARN=$((WARN+1))
fi

# ── 6. 检查 git 历史中是否有泄露的 key ──
echo -n "[6/7] No secrets in git history... "
HISTORY_LEAK=""
if git log --all -p 2>/dev/null | grep -qE "sk-[a-f0-9]{32}" 2>/dev/null; then
    HISTORY_LEAK="true"
fi
if [ -z "$HISTORY_LEAK" ]; then
    echo -e "${GREEN}PASS${NC}"
    PASS=$((PASS+1))
else
    echo -e "${RED}FAIL${NC} — Secret found in git history! Use git-filter-repo to clean."
    FAIL=$((FAIL+1))
fi

# ── 7. 检查是否有意外的大文件或二进制文件 ──
echo -n "[7/7] No large/binary files tracked... "
LARGE_FILES=""
for f in $(git ls-files); do
    if [ -f "$f" ]; then
        SIZE=$(wc -c < "$f" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt 1048576 ]; then  # > 1MB
            LARGE_FILES="${LARGE_FILES} ${f}(${SIZE}B)"
        fi
    fi
done
if [ -z "$LARGE_FILES" ]; then
    echo -e "${GREEN}PASS${NC}"
    PASS=$((PASS+1))
else
    echo -e "${YELLOW}WARN${NC} — Large files:${LARGE_FILES}"
    WARN=$((WARN+1))
fi

# ── 结果汇总 ──
echo ""
echo "======================================="
echo "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${WARN} warnings${NC}"
echo "======================================="

if [ $FAIL -gt 0 ]; then
    echo ""
    echo -e "${RED}DO NOT PUSH TO GITHUB until all failures are fixed!${NC}"
    echo ""
    echo "Common fixes:"
    echo "  API key leak:    Delete the key on provider's website, generate a new one"
    echo "  Personal path:   Replace with \$PROJECT_ROOT or generic placeholder"
    echo "  .env tracked:    git rm --cached .env && git commit"
    echo "  History leak:    pip install git-filter-repo && git filter-repo --invert-paths --path .env"
    exit 1
else
    echo ""
    echo -e "${GREEN}All clear! Safe to push to GitHub.${NC}"
    exit 0
fi
