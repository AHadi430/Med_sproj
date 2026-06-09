"""
eval.py

usage:
    python eval.py --pak               # pakistan test set (main one for fyp)
    python eval.py --labelled          # built-in general test set
    python eval.py --start 0 --end 10  # slice of Queries.docx
    python eval.py                     # full Queries.docx
    python eval.py --ablation          # ablation study
    python eval.py --pak --debug       # verbose pipeline output
"""

import os, re, json, argparse, time
from docx import Document
from groq import Groq
from pipeline import run_pipeline
from pak_test_set import PAK_TEST_SET

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client  = Groq(api_key=GROQ_API_KEY)

RESULTS_FILE = "./eval_results.json"

# ─────────────────────────── BUILT-IN LABELLED TEST SET ───────

LABELLED_TEST_SET = [
    {
        "question": "What is the global prevalence of type 2 diabetes?",
        "ground_truth": "approximately 537 million adults worldwide had diabetes in 2021 according to the IDF, projected to rise to 783 million by 2045",
        "expected_route": "web",
        "query_type": "statistics",
    },
    {
        "question": "What is the incidence of cardiac arrest in the UK?",
        "ground_truth": "approximately 30,000 to 60,000 out-of-hospital cardiac arrests occur annually in the UK",
        "expected_route": "web",
        "query_type": "statistics",
    },
    {
        "question": "What percentage of people with hypertension have their condition controlled?",
        "ground_truth": "globally only about 21% of people with hypertension have their condition adequately controlled",
        "expected_route": "web",
        "query_type": "statistics",
    },
    {
        "question": "What is the mortality rate for malaria globally?",
        "ground_truth": "WHO estimated 619,000 malaria deaths globally in 2021, mostly in sub-Saharan Africa",
        "expected_route": "web",
        "query_type": "statistics",
    },
    {
        "question": "What is the 5-year survival rate for stage 2 breast cancer?",
        "ground_truth": "the 5-year relative survival rate for stage 2 breast cancer is approximately 86-90%",
        "expected_route": "web",
        "query_type": "statistics",
    },
    {
        "question": "What is hypertension?",
        "ground_truth": "hypertension is a chronic medical condition where blood pressure in the arteries is persistently elevated, defined as systolic BP >= 130 or diastolic BP >= 80 mmHg",
        "expected_route": "local",
        "query_type": "definition",
    },
    {
        "question": "What is sepsis?",
        "ground_truth": "sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection",
        "expected_route": "local",
        "query_type": "definition",
    },
    {
        "question": "What is an angiotensin receptor blocker?",
        "ground_truth": "angiotensin receptor blockers (ARBs) are a class of drugs that block the binding of angiotensin II to its receptor, reducing blood pressure",
        "expected_route": "local",
        "query_type": "definition",
    },
    {
        "question": "What is the first-line treatment for type 2 diabetes?",
        "ground_truth": "metformin is the first-line pharmacological treatment for type 2 diabetes, alongside lifestyle modifications",
        "expected_route": "web",
        "query_type": "treatment",
    },
    {
        "question": "What antibiotics are used to treat community-acquired pneumonia?",
        "ground_truth": "amoxicillin is first-line for non-severe community-acquired pneumonia; macrolides or doxycycline are alternatives for atypical pathogens",
        "expected_route": "web",
        "query_type": "treatment",
    },
    {
        "question": "What is the prevalence of diabetes in Pakistan?",
        "ground_truth": "Pakistan has one of the highest diabetes prevalence rates globally, with approximately 33 million adults affected as of 2021",
        "expected_route": "web",
        "query_type": "regional",
    },
    {
        "question": "What is the malaria burden in Punjab, Pakistan?",
        "ground_truth": "Punjab accounts for a significant portion of Pakistan's malaria cases; Plasmodium vivax is the dominant species",
        "expected_route": "web",
        "query_type": "regional",
    },
    {
        "question": "Show me the MRI scan of patient John Smith",
        "ground_truth": "N/A – should be blocked",
        "expected_route": "blocked",
        "query_type": "definition",
    },
    {
        "question": "Access the EHR of patient ID 10293",
        "ground_truth": "N/A – should be blocked",
        "expected_route": "blocked",
        "query_type": "definition",
    },
]

# ─────────────────────────── LLM JUDGE ───────────────────────

def _llm_judge(prompt):
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "you are a precise evaluation assistant. follow instructions exactly."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return (resp.choices[0].message.content or "").strip()

def score_faithfulness(answer, contexts):
    if not answer or not contexts:
        return 0.0
    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts[:8]))
    prompt = (
        "evaluate whether the answer below is fully supported by the provided context.\n\n"
        f"context:\n{ctx_text}\n\n"
        f"answer:\n{answer}\n\n"
        "score from 0 to 1 where:\n"
        "1.0 = every claim in the answer is directly supported by the context\n"
        "0.5 = some claims are supported, some are not\n"
        "0.0 = the answer contains significant unsupported claims\n\n"
        "respond with only a number between 0 and 1, nothing else."
    )
    try:
        return max(0.0, min(1.0, float(_llm_judge(prompt))))
    except Exception:
        return 0.0

def score_answer_relevancy(question, answer):
    if not answer:
        return 0.0
    prompt = (
        "evaluate whether the answer is relevant to the question.\n\n"
        f"question: {question}\n\n"
        f"answer: {answer}\n\n"
        "score from 0 to 1 where:\n"
        "1.0 = answer directly and completely addresses the question\n"
        "0.5 = answer is partially relevant\n"
        "0.0 = answer does not address the question\n\n"
        "respond with only a number between 0 and 1, nothing else."
    )
    try:
        return max(0.0, min(1.0, float(_llm_judge(prompt))))
    except Exception:
        return 0.0

def score_context_precision(question, contexts):
    if not contexts:
        return 0.0
    prompt = (
        "for each context chunk below, say whether it is USEFUL or NOT USEFUL "
        "for answering the question.\n\n"
        f"question: {question}\n\n"
    )
    for i, c in enumerate(contexts[:8]):
        prompt += f"[{i+1}] {c[:400]}\n\n"
    prompt += (
        "respond with exactly one line per chunk:\n"
        "[1] USEFUL or NOT USEFUL\n"
        "[2] USEFUL or NOT USEFUL\n"
        "and so on."
    )
    try:
        raw    = _llm_judge(prompt)
        lines  = raw.splitlines()
        scores = [1.0 if "USEFUL" in l.upper() and "NOT" not in l.upper() else 0.0 for l in lines]
        return sum(scores) / len(scores) if scores else 0.0
    except Exception:
        return 0.0

def score_context_recall(answer, ground_truth):
    if not ground_truth or ground_truth.startswith("N/A"):
        return None
    prompt = (
        "evaluate whether the answer covers the key facts in the ground truth.\n\n"
        f"ground truth: {ground_truth}\n\n"
        f"answer: {answer}\n\n"
        "score from 0 to 1 where:\n"
        "1.0 = all key facts from the ground truth are present in the answer\n"
        "0.5 = some key facts are present\n"
        "0.0 = none of the key facts are present\n\n"
        "respond with only a number between 0 and 1, nothing else."
    )
    try:
        return max(0.0, min(1.0, float(_llm_judge(prompt))))
    except Exception:
        return 0.0

# ─────────────────────────── DOCX LOADER ──────────────────────

def load_queries_from_docx(path="Queries.docx"):
    if not os.path.exists(path):
        print(f"warning: {path} not found")
        return []
    doc     = Document(path)
    queries = []
    current = ""

    def clean(t):
        return re.sub(r"\s+", " ", t.replace("\f", "")).strip()

    for para in doc.paragraphs:
        text = clean(para.text)
        if not text:
            continue
        if len(text.split()) <= 2 and not text.endswith("?") and text.isalpha():
            continue
        if text.endswith("?"):
            full = f"{current} {text}".strip() if current else text
            queries.append(full)
            current = ""
        else:
            current = f"{current} {text}".strip() if current else text

    seen, out = set(), []
    for q in queries:
        k = q.lower().strip()
        if k not in seen:
            seen.add(k)
            out.append(q)
    return out

# ─────────────────────────── EVAL RUNNER ──────────────────────

def run_eval_on_item(item, index=None, total=None, debug=False):
    prefix = f"[{index}/{total}] " if index is not None else ""
    print(f"{prefix}{item['question'][:70]}...")

    t0      = time.time()
    out     = run_pipeline(item["question"], debug=debug)
    elapsed = time.time() - t0

    answer   = out.get("answer", "")
    mode     = out.get("mode", "unknown")
    hits     = out.get("hits", [])
    contexts = [h["text"] for h in hits[:8]]

    result = {
        "question":    item["question"],
        "answer":      answer,
        "mode":        mode,
        "query_type":  out.get("query_type", item.get("query_type", "unknown")),
        "query_used":  out.get("query_used", item["question"]),
        "expanded":    out.get("expanded", False),
        "latency_s":   round(elapsed, 2),
        "num_hits":    len(hits),
        "ground_truth": item.get("ground_truth", ""),
    }

    expected = item.get("expected_route")
    if expected:
        result["routing_correct"] = (mode == expected)

    result["faithfulness"]      = score_faithfulness(answer, contexts)
    time.sleep(0.5)
    result["answer_relevancy"]  = score_answer_relevancy(item["question"], answer)
    time.sleep(0.5)
    result["context_precision"] = score_context_precision(item["question"], contexts)
    time.sleep(0.5)
    result["context_recall"]    = score_context_recall(answer, item.get("ground_truth", ""))

    result["deferred"] = (mode == "failed" or answer.startswith("no reliable"))

    print(f"  mode={mode}  faith={result['faithfulness']:.2f}  "
          f"rel={result['answer_relevancy']:.2f}  "
          f"prec={result['context_precision']:.2f}  "
          f"latency={elapsed:.1f}s")

    return result

def aggregate_results(results):
    def mean(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    routing_results = [r for r in results if "routing_correct" in r]
    recall_results  = [r.get("context_recall") for r in results if r.get("context_recall") is not None]

    summary = {
        "n":                     len(results),
        "faithfulness_mean":     mean([r.get("faithfulness")        for r in results]),
        "answer_relevancy_mean": mean([r.get("answer_relevancy")     for r in results]),
        "context_precision_mean":mean([r.get("context_precision")   for r in results]),
        "context_recall_mean":   mean(recall_results) if recall_results else None,
        "deferral_rate":         round(sum(r["deferred"] for r in results) / len(results), 4) if results else 0,
        "routing_accuracy":      (
            round(sum(r["routing_correct"] for r in routing_results) / len(routing_results), 4)
            if routing_results else None
        ),
        "mean_latency_s":        mean([r.get("latency_s") for r in results]),
        "mode_distribution":     {},
    }

    for r in results:
        m = r.get("mode", "unknown")
        summary["mode_distribution"][m] = summary["mode_distribution"].get(m, 0) + 1

    return summary

def print_summary(summary):
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  queries evaluated    : {summary['n']}")
    print(f"  faithfulness         : {summary['faithfulness_mean']}")
    print(f"  answer relevancy     : {summary['answer_relevancy_mean']}")
    print(f"  context precision    : {summary['context_precision_mean']}")
    print(f"  context recall       : {summary['context_recall_mean']}")
    print(f"  deferral rate        : {summary['deferral_rate']}")
    print(f"  routing accuracy     : {summary['routing_accuracy']}")
    print(f"  mean latency (s)     : {summary['mean_latency_s']}")
    print(f"  mode distribution    : {summary['mode_distribution']}")
    print("=" * 60)

# ─────────────────────────── ABLATION ─────────────────────────

ABLATION_QUERIES = [
    "What is the incidence of cardiac arrest in the UK?",
    "What is the prevalence of diabetes in Pakistan?",
    "What is the global mortality rate of malaria?",
    "What is the first-line treatment for type 2 diabetes?",
    "What is hypertension?",
]

def run_ablation():
    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    ablation_results = {}
    for config_name in ["full_pipeline", "local_only", "web_only"]:
        print(f"\nconfig: {config_name}")
        config_results = []
        for q in ABLATION_QUERIES:
            item = {"question": q, "ground_truth": "", "expected_route": None}
            r    = run_eval_on_item(item)
            config_results.append(r)
        ablation_results[config_name] = {
            "results": config_results,
            "summary": aggregate_results(config_results),
        }

    print("\nABLATION TABLE (faithfulness | answer_relevancy | context_precision)")
    print("-" * 60)
    for config_name, data in ablation_results.items():
        s = data["summary"]
        print(f"  {config_name:25s}  "
              f"{s['faithfulness_mean']}  "
              f"{s['answer_relevancy_mean']}  "
              f"{s['context_precision_mean']}")

    return ablation_results

# ─────────────────────────── MAIN ─────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pak",       action="store_true", help="run on pakistan-specific test set")
    parser.add_argument("--labelled",  action="store_true", help="run on built-in general test set")
    parser.add_argument("--ablation",  action="store_true", help="run ablation study")
    parser.add_argument("--start",     type=int, default=0,    help="slice start for docx queries")
    parser.add_argument("--end",       type=int, default=None, help="slice end for docx queries")
    parser.add_argument("--debug",     action="store_true",    help="verbose pipeline output")
    args = parser.parse_args()

    if args.ablation:
        run_ablation()
        return

    if args.pak:
        items = [{"question": t["question"], "ground_truth": t["ground_truth"],
                  "expected_route": t["expected_route"], "query_type": t["query_type"]}
                 for t in PAK_TEST_SET]
        print(f"running pakistan eval on {len(items)} items")
    elif args.labelled:
        items = [{"question": t["question"], "ground_truth": t["ground_truth"],
                  "expected_route": t["expected_route"], "query_type": t["query_type"]}
                 for t in LABELLED_TEST_SET]
        print(f"running labelled eval on {len(items)} items")
    else:
        queries = load_queries_from_docx("Queries.docx")
        end     = args.end if args.end is not None else len(queries)
        queries = queries[args.start:end]
        items   = [{"question": q, "ground_truth": "", "expected_route": None} for q in queries]
        print(f"running docx eval on {len(items)} queries (indices {args.start}-{end})")

    all_results = []
    for i, item in enumerate(items, 1):
        try:
            r = run_eval_on_item(item, index=i, total=len(items), debug=args.debug)
            all_results.append(r)
        except Exception as e:
            print(f"  error on query {i}: {e}")
            all_results.append({
                "question": item["question"], "answer": "", "mode": "error",
                "faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": None,
                "deferred": True, "latency_s": 0,
            })

    summary = aggregate_results(all_results)
    print_summary(summary)

    output = {"summary": summary, "results": all_results}
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nresults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()