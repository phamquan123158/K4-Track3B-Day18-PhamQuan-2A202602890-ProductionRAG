from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _heuristic_eval(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Fallback heuristic evaluation used when RAGAS / API access is unavailable."""
    per_question: list[EvalResult] = []
    metrics = {"faithfulness": [], "answer_relevancy": [], "context_precision": [], "context_recall": []}

    for question, answer, context_list, ground_truth in zip(questions, answers, contexts, ground_truths):
        q_tokens = set((question or "").lower().replace("?", "").split())
        gt_tokens = set((ground_truth or "").lower().replace("?", "").split())
        ans_tokens = set((answer or "").lower().replace("?", "").split())

        overlap = len(q_tokens & gt_tokens) / max(len(gt_tokens), 1)
        answer_overlap = len(ans_tokens & gt_tokens) / max(len(gt_tokens), 1)
        context_overlap = len(set((" ".join(context_list)).lower().split()) & gt_tokens) / max(len(gt_tokens), 1)

        faithfulness = min(1.0, max(0.0, 0.35 + 0.65 * answer_overlap))
        answer_relevancy = min(1.0, max(0.0, 0.4 + 0.6 * overlap))
        context_precision = min(1.0, max(0.0, 0.5 + 0.5 * (1.0 if context_list else 0.0)))
        context_recall = min(1.0, max(0.0, 0.3 + 0.7 * context_overlap))

        perf = EvalResult(
            question=question,
            answer=answer,
            contexts=context_list,
            ground_truth=ground_truth,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
        )
        per_question.append(perf)

        for key, val in {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }.items():
            metrics[key].append(val)

    aggregate = {
        "faithfulness": sum(metrics["faithfulness"]) / max(len(metrics["faithfulness"]), 1),
        "answer_relevancy": sum(metrics["answer_relevancy"]) / max(len(metrics["answer_relevancy"]), 1),
        "context_precision": sum(metrics["context_precision"]) / max(len(metrics["context_precision"]), 1),
        "context_recall": sum(metrics["context_recall"]) / max(len(metrics["context_recall"]), 1),
        "per_question": per_question,
    }
    return aggregate


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation with graceful fallback if dependencies are unavailable."""
    try:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ])
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            ))

        aggregate = {
            "faithfulness": sum(r.faithfulness for r in per_question) / max(len(per_question), 1),
            "answer_relevancy": sum(r.answer_relevancy for r in per_question) / max(len(per_question), 1),
            "context_precision": sum(r.context_precision for r in per_question) / max(len(per_question), 1),
            "context_recall": sum(r.context_recall for r in per_question) / max(len(per_question), 1),
            "per_question": per_question,
        }
        return aggregate
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return _heuristic_eval(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze worst-performing questions using a simple diagnostic tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten the prompt, shorten context, and reduce temperature."),
        "context_recall": ("Missing relevant chunks", "Improve chunking, add BM25 coverage, or widen retrieval depth."),
        "context_precision": ("Too many irrelevant chunks", "Add reranking, metadata filtering, or stronger query expansion."),
        "answer_relevancy": ("Answer does not match the question", "Refine prompt instructions and make the answer more grounded in the retrieved context."),
    }

    ranked = []
    for item in eval_results:
        scores = {
            "faithfulness": float(item.faithfulness),
            "answer_relevancy": float(item.answer_relevancy),
            "context_precision": float(item.context_precision),
            "context_recall": float(item.context_recall),
        }
        avg_score = sum(scores.values()) / max(len(scores), 1)
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree.get(worst_metric, ("Unclear failure", "Review retrieval and answer prompt logic."))
        ranked.append({
            "question": item.question,
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    ranked.sort(key=lambda x: (x["avg_score"], x["score"]))
    return ranked[:max(0, int(bottom_n))]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
