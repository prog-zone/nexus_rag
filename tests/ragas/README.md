# RAGAS Evaluation Setup

## Install

```bash
pip install ragas langchain-openai
```

## Workflow

### 1. Generate a test dataset from a real case

```bash
python -m tests.ragas.generate_dataset \
  --case-id <your-case-uuid> \
  --output ragas_dataset.json \
  --num-questions 20
```

This fetches completed documents from the case, parses them via Unstructured,
and uses RAGAS + GPT-4o to generate realistic legal QA pairs.

### 2. Run the evaluation

```bash
pytest tests/ragas/test_evaluation.py -v --dataset ragas_dataset.json
```

To run only retrieval (faster, no generation cost):
```bash
pytest tests/ragas/test_evaluation.py -v -k "retrieval_only" --dataset ragas_dataset.json
```

## Metrics & Thresholds

| Metric             | Threshold | What it catches                          |
|--------------------|-----------|------------------------------------------|
| context_precision  | 0.75      | Noisy/irrelevant chunks being retrieved  |
| context_recall     | 0.70      | Missing relevant clauses                 |
| faithfulness       | 0.85      | Hallucinated content in answers          |
| answer_relevancy   | 0.75      | Off-topic or vague answers               |

Faithfulness is set highest because in legal work, a hallucinated clause
reference is a real liability.

## Tips

- Run `generate_dataset.py` against multiple case types (contracts, NDAs, etc.)
  to get diverse coverage.
- Keep your dataset JSON in version control so regressions are trackable.
- Run evaluation in CI on a small subset (5-10 questions) to keep it fast,
  and do full runs before releases.