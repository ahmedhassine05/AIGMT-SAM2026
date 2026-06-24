# GRL AIGMT

[![Paper](https://img.shields.io/badge/Paper-SAM2026-4A90D9?style=flat-square&logo=academia&logoColor=white)](https://www.scitepress.org/publishedPapers/2026/150140/pdf/)
[![LLM](https://img.shields.io/badge/LLM-ChatGPT%205.5-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com/)

Transforms a `.xgrl` / TGRL model file into an AI-enabled TGRL specification across three phases:
Phase 1 — AI Readiness Assessment, Phase 2 — AI Transformation Patterns, Phase 3 — LLM-Based TGRL Transformation.

## Requirements

- Python 3.8+
- An OpenAI API key

## Running the script

```bash
python GRL_AIGMT.py model.xgrl
```

Output files are written to the same directory as the input file.

Expected output files: `model_phase1_readiness.json`, `model_phase2_transformations.json`, and `model_phase3_transformed.xgrl`

---

## Setting up the API key

### Option 1 — System environment variable (recommended)

**Windows**
```cmd
setx OPENAI_API_KEY "your-key-here"
```
Restart your terminal after running this.

**macOS / Linux**
```bash
export OPENAI_API_KEY="your-key-here"
```
To make it permanent, add the line above to your `~/.bashrc`, `~/.zshrc`, or equivalent.

---

### Option 2 — Project `.env` file (fallback)

Requires `python-dotenv`:
```bash
pip install python-dotenv
```

Create a `.env` file in the same directory as the script:
```
OPENAI_API_KEY=your-key-here
```

> The system environment variable always takes priority over the `.env` file.
