# WikiFin-SEA 🌏💰

**A Wikipedia-Grounded Benchmark for Evaluating LLM Hallucinations in Financial Contexts Across Southeast Asian Languages**

> *arXiv preprint in preparation — Agnes Jeni Makay, UNSW Sydney (2026)*

---

## Overview

Large Language Models (LLMs) are increasingly deployed in financial services across Southeast Asia — yet no benchmark exists to measure how often they hallucinate financial facts in SEA languages.

**WikiFin-SEA** fills this gap. It is the first hallucination benchmark that combines:
- ✅ **Financial domain** (macroeconomics, banking, credit, markets)
- ✅ **Southeast Asian languages** (Indonesian, Thai, Vietnamese, Tagalog)
- ✅ **Wikipedia as structured ground truth**

---

## Benchmark Statistics

| Language | Articles | QA Pairs | Script |
|---|---|---|---|
| English (baseline) | 20 | ~160 | Latin |
| Indonesian | 20 | ~160 | Latin |
| Thai | 16 | ~128 | Thai |
| Vietnamese | 16 | ~128 | Latin |
| Tagalog | 15 | ~120 | Latin |
| **Total** | **87** | **~696** | — |

---

## Financial Subdomains

| Subdomain | Description |
|---|---|
| Macroeconomics | GDP, inflation, interest rates, monetary policy |
| Banking & Credit | Credit scores, BNPL, loans, Basel accords |
| Financial Markets | Stock indices, derivatives, foreign exchange |
| Economic Indicators | CPI, unemployment, trade balance |
| Financial Regulation | Central bank roles, AML, financial law |
| Personal Finance | Tax, insurance, budgeting |

---

## Models Evaluated

- GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Llama 3.1 70B (Meta)
- Mistral Large (Mistral AI)
- Qwen2.5 72B (Alibaba) — included for multilingual coverage

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/nesaugust/wikifin-sea
cd wikifin-sea

# 2. Install dependencies
pip install requests rapidfuzz pandas tqdm

# 3. Run extraction pipeline (English + Indonesian)
python wikifin_sea_pipeline.py

# 4. Set API key and run evaluation
export OPENAI_API_KEY=your_key_here
# Then uncomment evaluation step in main() and re-run
```

---

## Pipeline Architecture

```
Wikipedia API
     │
     ▼
Article Extraction (MediaWiki API)
     │
     ▼
Sentence Filtering (factual sentence detection)
     │
     ▼
QA Pair Generation (template-based)
     │
     ▼
wikifin_sea_dataset.csv
     │
     ▼
LLM Evaluation (5 models, zero-shot)
     │
     ├── Method A: Fuzzy Match (RapidFuzz token_set_ratio)
     └── Method B: LLM-as-Judge (GPT-4o)
          │
          ▼
     Hallucination Rate by Language / Subdomain / Model
```

---

## Output Files

| File | Description |
|---|---|
| `data/wikifin_sea_dataset.csv` | Full QA pairs with metadata |
| `data/results_gpt_4o.csv` | GPT-4o evaluation results |
| `data/wikifin_sea_stats.json` | Aggregated statistics |

---

## Research Questions

- **RQ1:** Do LLMs hallucinate more on financial facts in SEA languages vs English?
- **RQ2:** Which financial subdomains show the highest hallucination rates?
- **RQ3:** Does a model's multilingual training data affect SEA hallucination rates?

---

## Citation

```bibtex
@misc{makay2026wikifinsea,
  title   = {WikiFin-SEA: A Wikipedia-Grounded Benchmark for Evaluating 
             LLM Hallucinations in Financial Contexts Across Southeast Asian Languages},
  author  = {Makay, Agnes Jeni},
  year    = {2026},
  note    = {arXiv preprint (in preparation)},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## Author

**Agnes Jeni Makay**  
Master of Data Science and Decisions, UNSW Sydney  
[LinkedIn](https://www.linkedin.com/in/agnes makay) · [GitHub](https://github.com/nesaugust) · [Portfolio](https://share.streamlit.io/nesaugust)
