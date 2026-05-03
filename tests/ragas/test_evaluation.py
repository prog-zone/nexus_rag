"""
RAGAS evaluation tests for the legal RAG retrieval pipeline.

Run with:
    pytest tests/ragas/test_evaluation.py -v --dataset ragas_dataset.json

These tests evaluate:
- Context Precision   : Are retrieved chunks relevant to the question?
- Context Recall      : Does retrieval cover the answer's grounding?
- Faithfulness        : Is the answer grounded in retrieved context (no hallucination)?
- Answer Relevancy    : Does the answer actually address the question?
"""

import json
import pytest
from pathlib import Path
from openai import AsyncOpenAI

from ragas import evaluate
from ragas.metrics.collections import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset, EvaluationResult

from app.core.config import settings
from app.services.retrieval import retrieval_service
from app.services.llm import SYSTEM_PROMPT

# Legal domain: faithfulness is highest — hallucinated clauses are a liability.
THRESHOLDS = {
    "context_precision": 0.75,
    "context_recall":    0.70,
    "faithfulness":      0.85,
    "answer_relevancy":  0.75,
}


@pytest.fixture(scope="session")
def dataset_path(request):
    return Path(request.config.getoption("--dataset"))


@pytest.fixture(scope="session")
def qa_dataset(dataset_path):
    if not dataset_path.exists():
        pytest.skip(
            f"RAGAS dataset not found at {dataset_path}. "
            "Run generate_dataset.py first, or create it manually."
        )
    return json.loads(dataset_path.read_text())

async def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Call OpenAI directly using the same prompt as llm_service, without DB."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    context_text = "\n\n".join(
        f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Use only the following context to answer:\n\n{context_text}"},
        {"role": "user", "content": question},
    ]

    response = await client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,  # type: ignore
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""


async def run_pipeline(question: str, case_id: str) -> dict:
    """Run retrieval + generation for one question."""
    retrieval_result = await retrieval_service.retrieve(
        query=question,
        case_id=case_id,
        chat_id="ragas-eval-chat",
        user_id="ragas-eval-user",
    )

    chunks = retrieval_result.get("document_chunks", [])
    contexts = [c["text"] for c in chunks]

    if not contexts:
        return {"answer": "I could not find relevant information in the case documents.", "contexts": []}

    answer = await generate_answer(question, contexts)
    return {"answer": answer, "contexts": contexts}


def assert_scores(result: EvaluationResult, metrics: list[str]):
    """Assert all metric means meet their thresholds."""
    df = result.to_pandas()
    means = df.mean(numeric_only=True)

    print("\n── RAGAS Scores ──────────────────────────────")
    for metric in metrics:
        score = means.get(metric, 0)
        threshold = THRESHOLDS[metric]
        print(f"  {metric}: {score:.3f}  (threshold >= {threshold})")
        assert score >= threshold, (
            f"'{metric}' scored {score:.3f}, below threshold {threshold}. "
            "Review retrieval or generation quality."
        )

@pytest.mark.asyncio
@pytest.mark.ragas
async def test_ragas_pipeline_evaluation(qa_dataset, ragas_llm, ragas_embeddings):
    """End-to-end evaluation: retrieval + generation scored on all 4 metrics."""
    samples = []

    for item in qa_dataset:
        question = item["user_input"]
        ground_truth = item.get("reference", "")
        case_id = item.get("metadata", {}).get("case_id", "")

        if not case_id:
            continue

        result = await run_pipeline(question, case_id)

        samples.append(SingleTurnSample(
            user_input=question,
            reference=ground_truth,
            response=result["answer"],
            retrieved_contexts=result["contexts"],
        ))

    assert samples, "No samples produced — check your dataset has valid case_ids."

    scores = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        return_executor=False,  # Always returns EvaluationResult, never Executor
    )

    assert isinstance(scores, EvaluationResult)
    assert_scores(scores, ["context_precision", "context_recall", "faithfulness", "answer_relevancy"])


@pytest.mark.asyncio
@pytest.mark.ragas
async def test_ragas_retrieval_only(qa_dataset, ragas_llm, ragas_embeddings):
    """
    Retrieval-only evaluation (context precision + recall).
    Faster and cheaper — useful for isolating retrieval problems
    from generation problems.
    """
    samples = []

    for item in qa_dataset:
        question = item["user_input"]
        ground_truth = item.get("reference", "")
        case_id = item.get("metadata", {}).get("case_id", "")

        if not case_id:
            continue

        retrieval_result = await retrieval_service.retrieve(
            query=question,
            case_id=case_id,
            chat_id="ragas-eval-chat",
            user_id="ragas-eval-user",
        )
        contexts = [c["text"] for c in retrieval_result.get("document_chunks", [])]

        samples.append(SingleTurnSample(
            user_input=question,
            reference=ground_truth,
            response="",
            retrieved_contexts=contexts,
        ))

    assert samples

    scores = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        return_executor=False,
    )

    assert isinstance(scores, EvaluationResult)
    assert_scores(scores, ["context_precision", "context_recall"])