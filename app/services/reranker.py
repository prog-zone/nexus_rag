import voyageai
from app.core.config import settings
from app.core.logger import log


class RerankerService:
    def __init__(self):
        self.client = voyageai.AsyncClient(api_key=settings.VOYAGE_API_KEY)  # type: ignore
        self.model = settings.VOYAGE_RERANKER_MODEL
        self.instruction = (
            "Prioritize exact clause matches and specific legal terminology. "
            "Prefer documents that contain the exact legal entity mentioned in the query."
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 3
    ) -> list[dict]:
        
        if not documents:
            return []

        instructed_query = f"{self.instruction}\n\nQuery: {query}"

        result = await self.client.rerank(
            query=instructed_query,
            documents=documents,
            model=self.model,
            top_k=top_k,
        )

        reranked = []
        for item in result.results:
            reranked.append({
                "index": item.index,
                "text": documents[item.index],
                "score": item.relevance_score
            })

        log.info("reranking_complete", input=len(documents), output=len(reranked))
        return reranked


reranker_service = RerankerService()