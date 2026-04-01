# Contributing

Contributions are welcome! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/rdagent-quant-opt.git
cd rdagent-quant-opt
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Areas for Contribution

- **Adaptive routing**: Dynamically adjust model assignment based on observed success rates per stage
- **Batch API support**: Route non-urgent Implementation calls through DeepSeek's batch API for further cost savings
- **Fine-tuned models**: Train a small model specifically on Qlib factor code patterns
- **New scenarios**: Extend routing analysis to RD-Agent's Kaggle and medical prediction scenarios
- **Upstream PR**: Help integrate multi-model routing into RD-Agent's core codebase

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/adaptive-routing`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request with a clear description
