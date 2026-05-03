"""
Synthetic test dataset generator for legal RAG evaluation.

Usage:
    python -m tests.ragas.generate_dataset --case-id <uuid> --output dataset.json

This pulls real documents from a case via the retrieval service and uses
RAGAS testset generation to produce question/answer/context triples.
"""

import json
import asyncio
import argparse
from pathlib import Path

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document as LCDocument

from app.core.database import AsyncSessionLocal
from app.models.rag import Document, DocumentStatus
from app.services.s3 import s3_service
from app.services.unstructured import unstructured_service
from sqlalchemy import select


async def fetch_case_texts(case_id: str) -> list[LCDocument]:
    """Fetch and parse all completed documents for a case."""
    docs = []

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Document).where(
                Document.case_id == case_id,
                Document.status == DocumentStatus.COMPLETED,
            )
        )
        db_docs = result.scalars().all()

    for db_doc in db_docs:
        content = await s3_service.get_file_content(db_doc.s3_key)
        elements = await unstructured_service.partition_file_content(content, db_doc.filename) or []
        texts = [el["text"] for el in elements if "text" in el]

        for text in texts:
            docs.append(
                LCDocument(
                    page_content=text,
                    metadata={
                        "doc_id": str(db_doc.id),
                        "doc_name": db_doc.filename,
                        "case_id": case_id,
                    },
                )
            )

    return docs


async def generate(case_id: str, output_path: str, num_questions: int = 20):
    print(f"Fetching documents for case {case_id}...")
    documents = await fetch_case_texts(case_id)

    if not documents:
        raise ValueError("No completed documents found for this case.")

    print(f"Loaded {len(documents)} chunks. Generating {num_questions} QA pairs...")

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=num_questions,
    )

    df = testset.to_pandas()
    records = df.to_dict(orient="records")
    for record in records:
        record.setdefault("metadata", {})["case_id"] = case_id

    Path(output_path).write_text(json.dumps(records, indent=2))
    print(f"Dataset saved to {output_path} ({len(records)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", default="ragas_dataset.json")
    parser.add_argument("--num-questions", type=int, default=20)
    args = parser.parse_args()

    asyncio.run(generate(args.case_id, args.output, args.num_questions))