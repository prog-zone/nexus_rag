from pydantic import SecretStr
from qdrant_client import models
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.services.qdrant import qdrant_service
from app.services.embeddings import embedding_service
from app.core.logger import log

class RetrievalService:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=SecretStr(settings.OPENAI_API_KEY),
            model="gpt-4o-mini" 
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an intelligent assistant. Use the following retrieved context "
                "to answer the user's question. If the answer is not contained in the "
                "context, say 'I cannot answer this based on the provided document.' "
                "Do not make up information.\n\n"
                "Context:\n{context}"
            )),
            ("human", "{question}")
        ])

    async def get_answer(self, question: str, doc_id: str) -> str:
        """Searches Qdrant for relevant chunks and generates an LLM response."""
        
        dense_vecs, _ = embedding_service.generate_hybrid_embeddings([question])
        query_vector = dense_vecs[0].tolist()

        # Step 2: The Magic Filtered Search
        search_results = qdrant_service.client.search(
            collection_name=qdrant_service.collection_name,
            query_vector=("dense", query_vector),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=5
        )

        if not search_results:
            return "I couldn't find any relevant information in this document."

        context_chunks = [hit.payload.get("text", "") for hit in search_results if hit.payload]
        context_text = "\n\n---\n\n".join(context_chunks)

        chain = self.prompt | self.llm
        
        log.info("generating_llm_response", doc_id=doc_id)
        
        # Use ainvoke for async execution
        response = await chain.ainvoke({
            "context": context_text,
            "question": question
        })

        return response.content

retrieval_service = RetrievalService()