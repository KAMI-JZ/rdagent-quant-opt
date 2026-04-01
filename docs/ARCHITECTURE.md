# Architecture Analysis: RD-Agent(Q) Pipeline

> Reverse-engineered from source code (github.com/microsoft/RD-Agent) and paper (arXiv:2505.15155v2)

## Overview

RD-Agent(Q) decomposes quantitative R&D into five LLM-powered units operating in a closed loop. This document details the exact AI usage at each stage, based on source code analysis and paper appendix prompt templates.

## Critical Finding: Single-Model Architecture

RD-Agent uses a **single global `CHAT_MODEL` environment variable** for all LLM calls across all five modules. The backend (`rdagent.oai.backend.LiteLLMAPIBackend`) routes every request to the same model.

```
# From rdagent source:
CHAT_MODEL=gpt-4o          ← All 5 units share this
EMBEDDING_MODEL=text-embedding-ada-002  ← Co-STEER knowledge base
```

There is no built-in mechanism for per-module model selection.

## Per-Module Breakdown

### Module 1: Specification Unit

**Role**: Define constraints, data schemas, output formats, and the Qlib backtesting environment.

**AI Usage**: Near-zero. Primarily assembles pre-defined prompt templates. May make 0–1 LLM calls to dynamically adjust prompt phrasing based on the current optimization target.

**Token Consumption**: ~2,000 tokens/iteration (as part of system prompt injected into other modules).

### Module 2: Synthesis Unit ("The Researcher")

**Role**: Generate new factor/model hypotheses based on historical experiment results.

**AI Usage**:
- 1–2 LLM calls per iteration
- Input: SOTA factor list + full experiment trajectory (hypotheses + feedback pairs)
- Output: Structured hypothesis (name, economic rationale, formula, data fields, expected signal direction)

**Token Consumption**: 5,000–15,000 input tokens (grows with iteration count as history accumulates). Output ~500–1,500 tokens.

**Required AI Capability**: Creativity, financial domain knowledge, structured reasoning, diversity in hypothesis generation.

**Impact**: A poor hypothesis wastes the entire downstream pipeline (implementation + validation + analysis) for that iteration. No self-correction mechanism exists at this stage.

### Module 3: Implementation Unit / Co-STEER ("The Programmer")

**Role**: Translate hypotheses into runnable Python code (Qlib-compatible).

**AI Usage** (three nested loops):

1. **Scheduling loop** (1 LLM call): Build task dependency DAG, determine execution order
2. **Code generation loop** (1 call per subtask): Generate Python code from task description + knowledge base examples
3. **Debug loop** (1–10 calls per subtask): Execute code, feed errors back to LLM for fixes

**Embedding calls**: 1–3 per subtask for knowledge base similarity search (text-embedding-ada-002)

**Token Consumption per iteration**: ~30,000–60,000 input tokens total across all calls (2,000–4,000 per individual call).

**Total calls per iteration**: Typically 10–15 LLM calls + 3–9 embedding calls.

**Self-correction**: Yes — Co-STEER retries failed code up to 10 times (600s timeout per task). This is the critical insight: even if code generation quality is lower, the retry mechanism compensates.

**Required AI Capability**: Python code generation, Pandas/NumPy/Qlib API knowledge, error diagnosis. Tasks are highly structured with clear input/output specifications.

### Module 4: Validation Unit

**Role**: De-duplicate factors (IC correlation filtering), execute Qlib backtesting.

**AI Usage**: Essentially zero. All operations are mathematical (correlation computation) or programmatic (Qlib API calls). Timeout: 3,600 seconds per validation run.

### Module 5: Analysis Unit ("The Strategy Director")

**Role**: Evaluate experiment results against SOTA, diagnose failures, generate improvement suggestions, and schedule next optimization direction (factor vs. model) via Thompson Sampling bandit.

**AI Usage**:
- 1–2 LLM calls per iteration
- Input: Experiment metrics (IC, ICIR, ARR, MDD, etc.) + hypothesis description
- Output: Qualitative diagnosis + suggested improvements + direction recommendation

**Token Consumption**: ~3,000–5,000 input tokens, ~500–1,000 output.

**Note**: The Multi-Armed Bandit scheduler itself is a pure mathematical algorithm (Thompson Sampling), NOT an LLM call.

**Required AI Capability**: Quantitative metric interpretation, causal reasoning, strategic judgment.

**Impact**: Like Synthesis, there is no self-correction mechanism. A bad analysis propagates misinformation to the next Synthesis step.

## Aggregate Cost Distribution

For a typical 30-iteration run:

| Module | LLM Calls | % of Total Calls | % of Token Budget | Self-Correction |
|:---|---:|---:|---:|:---:|
| Specification | 0–30 | ~2% | ~3% | N/A |
| Synthesis | 30–60 | ~10% | ~15% | ❌ |
| **Implementation** | **300–450** | **~70%** | **~65%** | **✅ (10× retry)** |
| Validation | 0 | 0% | 0% | N/A |
| Analysis | 30–60 | ~10% | ~12% | ❌ |
| Embedding | 90–270 | ~8% | ~5% | N/A |

## Implications for Multi-Model Routing

The cost distribution reveals a clear optimization opportunity:

1. **Implementation dominates cost but tolerates weaker models** due to self-correction
2. **Synthesis and Analysis are cheap but intolerant of quality drops** (no retry mechanism)
3. **Validation costs zero in LLM terms** (pure computation)

This maps naturally to a two-tier routing strategy:
- Tier 1 (Strong): Synthesis + Analysis — low frequency, high impact
- Tier 2 (Efficient): Implementation + Specification — high frequency, self-correcting

## Paper Experimental Evidence

The paper tested RD-Agent with different LLM backends (Figure 9, 30 iterations each):

| Backend | IC | ARR | Notes |
|:---|:---:|:---:|:---|
| o3-mini | 0.0532 | 14.21% | Best overall |
| GPT-4.1 | Good | Good | Second best |
| GPT-4o | 0.0475 | ~12% | Moderate |
| **GPT-4o-mini** | **Poor** | **Poor** | **Worst — too weak for Synthesis/Analysis** |
| o1 | Special | Special | Fewer valid loops but high per-loop quality |

The GPT-4o-mini result is particularly telling: it demonstrates that a globally weak model hurts Synthesis and Analysis quality, which cascades into poor overall performance — even though Implementation (with its self-correction) might have worked adequately.
