"""
WikiFin-SEA: Wikipedia-Grounded Financial Hallucination Benchmark
ETL Pipeline for QA Pair Extraction + Evaluation

Author: Agnes Jeni Makay
Project: WikiFin-SEA arXiv Preprint

SETUP:
    pip install requests rapidfuzz pandas tqdm

API KEYS (both free, no credit card needed):
    Groq:            https://console.groq.com  → free, sign up with email
    Google AI Studio: https://aistudio.google.com → free, sign in with Google account

USAGE (PowerShell):
    $env:GROQ_API_KEY = "gsk_your_key_here"
    $env:GOOGLE_API_KEY = "your_google_ai_studio_key_here"
    python wikifin_sea_pipeline.py

OUTPUT:
    data/wikifin_sea_dataset.csv                        full QA dataset
    data/results_llama_3_3_70b_versatile.csv            Groq / Llama 3.3 70B results
    data/results_gemini_2_5_flash.csv                   Google / Gemini 2.5 Flash results
    data/comparison_summary.csv                         side by side model comparison
    data/wikifin_sea_stats.json                         dataset statistics
"""

import requests
import pandas as pd
import json
import re
import time
import os
from tqdm import tqdm
from rapidfuzz import fuzz

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

LANGUAGES = {
    "en": "English",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
    "tl": "Tagalog",
}

FINANCIAL_TOPICS = {
    "en": [
        "Gross domestic product", "Inflation", "Interest rate", "Monetary policy",
        "Central bank", "Credit score", "Buy now pay later", "Basel III",
        "Stock market index", "Foreign exchange market", "Consumer price index",
        "Unemployment", "Balance of trade", "Anti-money laundering",
        "Income tax", "Insurance", "Bond (finance)", "Dividend",
        "Microfinance", "Financial inclusion"
    ],
    "id": [
        "Produk domestik bruto", "Inflasi", "Suku bunga", "Kebijakan moneter",
        "Bank sentral", "Skor kredit", "Beli sekarang bayar nanti", "Bursa saham",
        "Valuta asing", "Indeks harga konsumen", "Pengangguran",
        "Neraca perdagangan", "Pencucian uang", "Pajak penghasilan",
        "Asuransi", "Obligasi", "Dividen", "Keuangan mikro",
        "Bank Indonesia", "Otoritas Jasa Keuangan"
    ],
    "th": [
        "ผลิตภัณฑ์มวลรวมในประเทศ", "อัตราเงินเฟ้อ", "อัตราดอกเบี้ย",
        "นโยบายการเงิน", "ธนาคารกลาง", "ตลาดหลักทรัพย์",
        "อัตราแลกเปลี่ยน", "ดัชนีราคาผู้บริโภค", "การว่างงาน",
        "ดุลการค้า", "ภาษีเงินได้", "ประกันภัย", "พันธบัตร",
        "เงินปันผล", "สินเชื่อรายย่อย", "ธนาคารแห่งประเทศไทย"
    ],
    "vi": [
        "Tổng sản phẩm quốc nội", "Lạm phát", "Lãi suất",
        "Chính sách tiền tệ", "Ngân hàng trung ương", "Thị trường chứng khoán",
        "Tỷ giá hối đoái", "Chỉ số giá tiêu dùng", "Thất nghiệp",
        "Cán cân thương mại", "Thuế thu nhập", "Bảo hiểm",
        "Trái phiếu", "Cổ tức", "Tài chính vi mô", "Ngân hàng Nhà nước Việt Nam"
    ],
    "tl": [
        "Kabuuang domestic na produkto", "Implasyon", "Tubo",
        "Patakaran sa pananalapi", "Sentral na bangko", "Merkado ng stock",
        "Palitan ng pera", "Indeks ng presyo ng consumer", "Kawalan ng trabaho",
        "Balanse ng kalakalan", "Buwis sa kita", "Seguro",
        "Bono", "Dibidendo", "Bangko Sentral ng Pilipinas"
    ]
}

SUBDOMAIN_KEYWORDS = {
    "Macroeconomics": ["gdp", "produk domestik", "gross domestic", "inflation", "inflasi",
                       "interest rate", "suku bunga", "monetary policy", "kebijakan moneter",
                       "lạm phát", "lãi suất", "อัตราเงินเฟ้อ", "อัตราดอกเบี้ย"],
    "Banking & Credit": ["credit", "kredit", "bank", "loan", "pinjaman", "bnpl", "buy now pay later",
                         "beli sekarang", "basel", "microfinance", "keuangan mikro"],
    "Financial Markets": ["stock", "saham", "bursa", "chứng khoán", "หลักทรัพย์", "bond", "obligasi",
                          "dividend", "dividen", "foreign exchange", "valuta asing"],
    "Economic Indicators": ["gdp", "cpi", "consumer price", "harga konsumen", "unemployment",
                            "pengangguran", "trade balance", "neraca perdagangan"],
    "Financial Regulation": ["central bank", "bank sentral", "anti-money laundering", "pencucian uang",
                             "basel", "otoritas", "regulation", "regulasi"],
    "Personal Finance": ["tax", "pajak", "insurance", "asuransi", "income", "pendapatan",
                         "bảo hiểm", "ประกัน"]
}

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── MODELS TO EVALUATE ───────────────────────────────────────────────────────
# Both are free with no credit card needed.
#
# Model 1: Groq / Llama 3.3 70B
#   Sign up at https://console.groq.com (email only, no card)
#   Set: $env:GROQ_API_KEY = "gsk_..."
#
# Model 2: Google AI Studio / Gemini 2.5 Flash
#   Sign in at https://aistudio.google.com with your Google account
#   Get key at https://aistudio.google.com/app/apikey
#   Set: $env:GOOGLE_API_KEY = "AIza..."
# ─────────────────────────────────────────────────────────────────────────────
EVALUATION_MODELS = [
    {
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "label": "Llama 3.3 70B (Groq)",
        "delay_between_pairs": 4,
        "cooldown_every_n": 50,
        "cooldown_seconds": 30,
    },
    {
        "model": "gemini-2.5-flash",
        "provider": "google",
        "label": "Gemini 2.5 Flash (Google AI Studio)",
        "delay_between_pairs": 2,
        "cooldown_every_n": 100,
        "cooldown_seconds": 15,
    },
]


# ─────────────────────────────────────────────
# 2. WIKIPEDIA API EXTRACTION
# ─────────────────────────────────────────────

def fetch_wikipedia_article(title: str, lang: str) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "titles": title, "prop": "extracts",
        "exintro": False, "explaintext": True, "exsectionformat": "plain",
        "redirects": True, "format": "json"
    }
    try:
        headers = {"User-Agent": "WikiFin-SEA-Benchmark/1.0 (agnes.makay@research.unsw.edu.au)"}
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        if "missing" in page:
            return None
        return {"title": page.get("title", title), "lang": lang, "extract": page.get("extract", "")}
    except Exception as e:
        print(f"  ⚠️  Failed to fetch '{title}' [{lang}]: {e}")
        return None


# ─────────────────────────────────────────────
# 3. SENTENCE EXTRACTION & FILTERING
# ─────────────────────────────────────────────

def split_into_sentences(text: str, lang: str) -> list:
    if not text:
        return []
    text = re.sub(r"==+[^=]+=+\n?", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\([^)]{0,50}\)", "", text)
    if lang == "th":
        sentences = [s.strip() for s in text.split("\n") if len(s.strip()) > 40]
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def is_factual_sentence(sentence: str) -> bool:
    if len(sentence.split()) < 6:
        return False
    skip_patterns = [
        r"^(this|it|they|he|she) (is|are|was|were) (also|often|sometimes|usually)",
        r"^(see also|references|external links|notes|bibliography)",
        r"^\d+\.",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, sentence.lower()):
            return False
    has_number = bool(re.search(r'\d', sentence))
    has_currency = bool(re.search(r'(USD|IDR|THB|VND|PHP|\$|Rp|฿|₫|₱)', sentence))
    has_financial_keyword = bool(re.search(
        r'(bank|rate|percent|inflation|gdp|growth|economy|financial|credit|market|'
        r'suku bunga|inflasi|persen|ekonomi|kredit|pasar|'
        r'ธนาคาร|เศรษฐกิจ|อัตรา|ngân hàng|kinh tế|lãi suất|bangko|ekonomiya)',
        sentence, re.IGNORECASE
    ))
    return has_number or has_currency or has_financial_keyword


# ─────────────────────────────────────────────
# 4. QA PAIR GENERATION
# ─────────────────────────────────────────────

QA_TEMPLATES = {
    "en": [
        ("What is {entity}?", "{sentence}"),
        ("What is the role of {entity}?", "{sentence}"),
        ("According to Wikipedia, what is notable about {entity}?", "{sentence}"),
    ],
    "id": [
        ("Apa itu {entity}?", "{sentence}"),
        ("Apa peran {entity}?", "{sentence}"),
        ("Menurut Wikipedia, apa yang penting tentang {entity}?", "{sentence}"),
    ],
    "th": [("{entity} คืออะไร?", "{sentence}"), ("บทบาทของ {entity} คืออะไร?", "{sentence}")],
    "vi": [("{entity} là gì?", "{sentence}"), ("Vai trò của {entity} là gì?", "{sentence}")],
    "tl": [("Ano ang {entity}?", "{sentence}"), ("Ano ang papel ng {entity}?", "{sentence}")],
}


def generate_qa_pairs(article: dict, max_pairs: int = 10) -> list:
    lang = article["lang"]
    title = article["title"]
    sentences = split_into_sentences(article["extract"], lang)
    factual_sentences = [s for s in sentences if is_factual_sentence(s)][:max_pairs]
    templates = QA_TEMPLATES.get(lang, QA_TEMPLATES["en"])
    qa_pairs = []
    for i, sentence in enumerate(factual_sentences):
        template_q, _ = templates[i % len(templates)]
        qa_pairs.append({
            "id": f"{lang}_{re.sub(r'[^a-z0-9]', '_', title.lower())}_{i:03d}",
            "language": lang,
            "language_name": LANGUAGES[lang],
            "article_title": title,
            "subdomain": label_subdomain(title + " " + sentence),
            "question": template_q.format(entity=title),
            "reference_answer": sentence,
            "source_url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        })
    return qa_pairs


def label_subdomain(text: str) -> str:
    text_lower = text.lower()
    for domain, keywords in SUBDOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    return "General Finance"


# ─────────────────────────────────────────────
# 5. LLM QUERYING
# ─────────────────────────────────────────────

PROMPTS = {
    "en": "Answer the following question concisely and factually:\n\nQuestion: {question}\nAnswer:",
    "id": "Jawab pertanyaan berikut secara singkat dan faktual:\n\nPertanyaan: {question}\nJawaban:",
    "th": "ตอบคำถามต่อไปนี้อย่างกระชับและถูกต้อง:\n\nคำถาม: {question}\nคำตอบ:",
    "vi": "Trả lời câu hỏi sau một cách ngắn gọn và chính xác:\n\nCâu hỏi: {question}\nTrả lời:",
    "tl": "Sagutin ang sumusunod na tanong nang maikli at tumpak:\n\nTanong: {question}\nSagot:",
}


def query_llm(question: str, lang: str, model: str, provider: str) -> str:
    """
    Route to the correct provider.

    FREE PROVIDERS (no credit card):
      groq   — https://console.groq.com      GROQ_API_KEY
      google — https://aistudio.google.com   GOOGLE_API_KEY

    FUTURE PAID PROVIDERS (deep research phase):
      openai    — OPENAI_API_KEY
      anthropic — ANTHROPIC_API_KEY
    """
    prompt = PROMPTS.get(lang, PROMPTS["en"]).format(question=question)

    if provider == "google":
        return _query_google(prompt, model)
    if provider == "anthropic":
        return _query_anthropic(prompt, model)

    OPENAI_COMPAT = {
        "groq":     {"url": "https://api.groq.com/openai/v1/chat/completions",    "key": os.getenv("GROQ_API_KEY", "")},
        "openai":   {"url": "https://api.openai.com/v1/chat/completions",         "key": os.getenv("OPENAI_API_KEY", "")},
        "together": {"url": "https://api.together.xyz/v1/chat/completions",       "key": os.getenv("TOGETHER_API_KEY", "")},
    }
    config = OPENAI_COMPAT.get(provider)
    if not config or not config["key"]:
        print(f"  ⚠️  No API key for provider '{provider}'. Set {provider.upper()}_API_KEY.")
        return ""
    try:
        r = requests.post(
            config["url"],
            headers={"Authorization": f"Bearer {config['key']}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 150, "temperature": 0.0},
            timeout=30
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠️  LLM query failed ({provider}/{model}): {e}")
        return ""


def _query_google(prompt: str, model: str) -> str:
    """Google AI Studio (Gemini) via REST. Free, no credit card."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        print("  ⚠️  GOOGLE_API_KEY not set. Get one free at https://aistudio.google.com/app/apikey")
        return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": 150, "temperature": 0.0}},
            timeout=30
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"  ⚠️  Google query failed ({model}): {e}")
        return ""


def _query_anthropic(prompt: str, model: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ⚠️  ANTHROPIC_API_KEY not set.")
        return ""
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": model, "max_tokens": 150, "messages": [{"role": "user", "content": prompt}]},
            timeout=30
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
    except Exception as e:
        print(f"  ⚠️  Anthropic query failed: {e}")
        return ""


# ─────────────────────────────────────────────
# 6. HALLUCINATION DETECTION
# ─────────────────────────────────────────────

def detect_hallucination_fuzzy(model_answer: str, reference_answer: str, threshold: int = 60) -> tuple:
    """Method A: Fuzzy string matching. Below threshold = hallucinated."""
    if not model_answer or not reference_answer:
        return True, 0.0
    score = fuzz.token_set_ratio(model_answer.lower().strip(), reference_answer.lower().strip())
    return score < threshold, float(score)


def detect_hallucination_llm_judge(question: str, reference: str, model_answer: str) -> tuple:
    """
    Method B: LLM as Judge.
    Always uses Groq / Llama 3.3 70B as the judge for consistency across target models.
    """
    judge_prompt = f"""You are a factual consistency judge evaluating an AI answer against a Wikipedia reference.

Reference answer (from Wikipedia): "{reference}"
AI model answer: "{model_answer}"

Task: Is the AI model answer factually consistent with the reference?
- Paraphrasing is acceptable, focus only on factual accuracy
- If the model answer contains correct facts (even if worded differently), answer YES
- If the model answer contains incorrect or fabricated facts, answer NO
- Answer ONLY with YES or NO on the first line, then one sentence explaining why.

Response:"""

    judge_text = query_llm(judge_prompt, lang="en", model="llama-3.3-70b-versatile", provider="groq")
    if not judge_text:
        return None, "Judge query failed"
    return judge_text.strip().upper().startswith("NO"), judge_text


# ─────────────────────────────────────────────
# 7. PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────

def run_extraction_pipeline(languages=None, max_pairs_per_article=8, delay=1.0) -> pd.DataFrame:
    """Step 1: Extract Wikipedia articles and generate QA pairs."""
    if languages is None:
        languages = list(LANGUAGES.keys())
    all_qa_pairs = []
    for lang in languages:
        topics = FINANCIAL_TOPICS.get(lang, [])
        print(f"\n📖 Extracting [{LANGUAGES[lang]}] — {len(topics)} topics...")
        for topic in tqdm(topics, desc=f"  {lang}"):
            article = fetch_wikipedia_article(topic, lang)
            if article and article["extract"]:
                all_qa_pairs.extend(generate_qa_pairs(article, max_pairs=max_pairs_per_article))
            time.sleep(delay)
    df = pd.DataFrame(all_qa_pairs)
    out = os.path.join(OUTPUT_DIR, "wikifin_sea_dataset.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✅ Dataset saved: {out} ({len(df)} QA pairs)")
    return df


def run_evaluation_pipeline(df: pd.DataFrame, model: str, provider: str, label: str,
                             delay_between_pairs=4, cooldown_every_n=50,
                             cooldown_seconds=30, sample_size=None) -> pd.DataFrame:
    """Step 2: Query one LLM and detect hallucinations using Method A + B."""
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42).copy()
    else:
        df = df.copy()

    print(f"\n🤖 Evaluating [{label}] on {len(df)} QA pairs...")
    print(f"   Method A: Fuzzy Match | Method B: LLM as Judge (Groq / Llama 3.3 70B)")
    est = len(df) * (delay_between_pairs + 2)
    print(f"   Estimated time: ~{est // 60} min {est % 60} sec\n")

    model_answers, method_a_flags, hallucinated_flags, scores, judges = [], [], [], [], []

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):

        if i > 0 and i % cooldown_every_n == 0:
            print(f"\n  ⏸️  Cooldown after {i} pairs — waiting {cooldown_seconds}s...")
            time.sleep(cooldown_seconds)

        answer = ""
        for attempt in range(5):
            answer = query_llm(row["question"], row["language"], model, provider)
            if answer:
                break
            wait = 10 * (attempt + 1)
            print(f"\n  ⏳ Rate limited — waiting {wait}s (retry {attempt + 1}/5)...")
            time.sleep(wait)

        is_hall_a, score = detect_hallucination_fuzzy(answer, row["reference_answer"])
        time.sleep(2)
        is_hall_b, judge_text = detect_hallucination_llm_judge(
            row["question"], row["reference_answer"], answer
        )

        is_hall_final = is_hall_a if is_hall_b is None else (is_hall_a and is_hall_b)

        model_answers.append(answer)
        method_a_flags.append(is_hall_a)
        hallucinated_flags.append(is_hall_final)
        scores.append(score)
        judges.append(judge_text)
        time.sleep(delay_between_pairs)

    df["model"] = model
    df["provider"] = provider
    df["model_answer"] = model_answers
    df["hallucinated_method_a"] = method_a_flags
    df["hallucinated"] = hallucinated_flags
    df["hallucination_score"] = scores
    df["judge_response"] = judges

    safe = model.replace("-", "_").replace("/", "_").replace(".", "_")
    out = os.path.join(OUTPUT_DIR, f"results_{safe}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    rate_a = pd.Series(method_a_flags).mean()
    rate_f = pd.Series(hallucinated_flags).mean()
    print(f"\n📊 [{label}] Results:")
    print(f"   Method A (fuzzy):      {rate_a:.1%}")
    print(f"   Final (A + B):         {rate_f:.1%}")
    print(f"   Judge correction:      {rate_a - rate_f:.1%} reduction")
    print(f"✅ Saved: {out}")
    return df


def generate_comparison_summary(result_dfs: list) -> pd.DataFrame:
    """Step 3b: Side by side comparison across models."""
    rows = []
    for df in result_dfs:
        m = df["model"].iloc[0]
        p = df["provider"].iloc[0] if "provider" in df.columns else "unknown"
        for lang, grp in df.groupby("language_name"):
            rows.append({"model": m, "provider": p, "language": lang,
                         "n_pairs": len(grp),
                         "method_a_rate": round(grp["hallucinated_method_a"].mean(), 4),
                         "final_rate": round(grp["hallucinated"].mean(), 4),
                         "avg_fuzzy_score": round(grp["hallucination_score"].mean(), 2)})
        rows.append({"model": m, "provider": p, "language": "OVERALL",
                     "n_pairs": len(df),
                     "method_a_rate": round(df["hallucinated_method_a"].mean(), 4),
                     "final_rate": round(df["hallucinated"].mean(), 4),
                     "avg_fuzzy_score": round(df["hallucination_score"].mean(), 2)})
    summary = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n📊 Comparison summary saved: {out}")
    print(summary[summary["language"] == "OVERALL"].to_string(index=False))
    return summary


def generate_statistics(df: pd.DataFrame) -> dict:
    stats = {
        "total_qa_pairs": len(df),
        "by_language_count": df.groupby("language_name").size().to_dict(),
        "by_subdomain_count": df.groupby("subdomain").size().to_dict(),
    }
    if "hallucinated" in df.columns and df["hallucinated"].notna().any():
        stats["overall_hallucination_rate"] = float(df["hallucinated"].mean())
        stats["by_language"] = df.groupby("language_name")["hallucinated"].mean().to_dict()
        stats["by_subdomain"] = df.groupby("subdomain")["hallucinated"].mean().to_dict()
        if "model" in df.columns:
            stats["by_model"] = df.groupby("model")["hallucinated"].mean().to_dict()
        if "hallucinated_method_a" in df.columns:
            stats["method_a_rate"] = float(df["hallucinated_method_a"].mean())
    out = os.path.join(OUTPUT_DIR, "wikifin_sea_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Statistics saved: {out}")
    return stats


# ─────────────────────────────────────────────
# 8. MAIN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("WikiFin-SEA ETL Pipeline")
    print("Wikipedia-Grounded Financial Hallucination Benchmark")
    print("=" * 60)

    ACTIVE_LANGUAGES = ["en", "id", "th", "vi", "tl"]

    # Step 1: Load existing dataset or extract fresh
    dataset_path = os.path.join(OUTPUT_DIR, "wikifin_sea_dataset.csv")
    if os.path.exists(dataset_path):
        print(f"\n📂 Loading existing dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"   {len(df)} QA pairs loaded.")
    else:
        df = run_extraction_pipeline(languages=ACTIVE_LANGUAGES, max_pairs_per_article=8, delay=1.0)

    if df.empty:
        print("  ⚠️  No QA pairs found. Check internet connection.")
        exit(1)

    # Step 2: Evaluate each model (skips if results CSV already exists)
    result_dfs = []
    for cfg in EVALUATION_MODELS:
        safe = cfg["model"].replace("-", "_").replace("/", "_").replace(".", "_")
        result_path = os.path.join(OUTPUT_DIR, f"results_{safe}.csv")

        if os.path.exists(result_path):
            print(f"\n📂 Loading existing results for [{cfg['label']}]: {result_path}")
            result_dfs.append(pd.read_csv(result_path))
            continue

        key_map = {"groq": "GROQ_API_KEY", "google": "GOOGLE_API_KEY",
                   "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        key_var = key_map.get(cfg["provider"], f"{cfg['provider'].upper()}_API_KEY")
        if not os.getenv(key_var):
            print(f"\n⚠️  Skipping [{cfg['label']}] — {key_var} not set.")
            print(f"   Set it with:  $env:{key_var} = 'your_key_here'")
            continue

        result_dfs.append(run_evaluation_pipeline(
            df, model=cfg["model"], provider=cfg["provider"], label=cfg["label"],
            delay_between_pairs=cfg["delay_between_pairs"],
            cooldown_every_n=cfg["cooldown_every_n"],
            cooldown_seconds=cfg["cooldown_seconds"],
        ))

    # Step 3: Statistics and comparison
    if result_dfs:
        stats = generate_statistics(pd.concat(result_dfs, ignore_index=True))
        if len(result_dfs) > 1:
            generate_comparison_summary(result_dfs)

    print("\n🎉 Pipeline complete!")
    for cfg in EVALUATION_MODELS:
        safe = cfg["model"].replace("-", "_").replace("/", "_").replace(".", "_")
        print(f"   Results: data/results_{safe}.csv")
    if len(result_dfs) > 1:
        print("   Comparison: data/comparison_summary.csv")