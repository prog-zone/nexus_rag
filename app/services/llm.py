import uuid
import json
from typing import AsyncGenerator
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.logger import log
from app.models.rag import Chat, ChatMessage, MessageRole


SYSTEM_PROMPT = """You are Nexus, a precise legal AI assistant. You help lawyers and legal teams analyze case documents with maximum accuracy.
        STRICT RULES:
        1. Only answer from the provided [Retrieved Context]. Never use outside knowledge for legal facts.
        2. Every claim MUST cite its source: [Source: document_name]
        3. If the answer is not in the context, say exactly: "I could not find this information in the case documents. Please verify the source document directly."
        4. If the query is ambiguous, ask the user to clarify which document or clause they mean.
        5. Never guess, infer, or hallucinate legal information.
        6. Be precise and concise. Legal accuracy is more important than length."""


class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_CHAT_MODEL

    def _build_messages(
        self,
        context_prompt: str,
        recent_messages: list[ChatMessage],
        user_query: str
    ) -> list[dict]:

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context_prompt:
            messages.append({
                "role": "system",
                "content": f"Use only the following context to answer:\n\n{context_prompt}"
            })

        for msg in recent_messages:
            if msg.role == MessageRole.SYSTEM:
                continue
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })

        messages.append({"role": "user", "content": user_query})
        return messages

    async def stream_response(
        self,
        query: str,
        context_prompt: str,
        recent_messages: list[ChatMessage],
        has_relevant_context: bool,
        chat: Chat,
        user_message: ChatMessage,
        db: AsyncSession
    ) -> AsyncGenerator[str, None]:

        if not has_relevant_context:
            failure_msg = (
                "I could not find relevant information in the case documents "
                "for your query. Please verify the source document directly "
                "or rephrase your question."
            )
            yield f"data: {json.dumps({'type': 'content', 'text': failure_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            await self._save_assistant_message(
                content=failure_msg,
                chat=chat,
                user_message=user_message,
                db=db
            )
            return

        full_response = ""
        try:
            messages = self._build_messages(
                context_prompt=context_prompt,
                recent_messages=recent_messages,
                user_query=query
            )

            log.info(
                "llm_request_started",
                model=self.model,
                message_count=len(messages),
                has_context=has_relevant_context
            )

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                stream=True,
                temperature=0.1,
                max_tokens=2048
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield f"data: {json.dumps({'type': 'content', 'text': delta})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            log.info("llm_stream_complete", chat_id=str(chat.id), tokens=len(full_response))

        except Exception as e:
            log.exception("llm_streaming_failed", chat_id=str(chat.id), error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'text': 'An error occurred generating the response.'})}\n\n"

        finally:
            if full_response:
                await self._save_assistant_message(
                    content=full_response,
                    chat=chat,
                    user_message=user_message,
                    db=db
                )

    async def _save_assistant_message(
        self,
        content: str,
        chat: Chat,
        user_message: ChatMessage,
        db: AsyncSession
    ):
        try:
            assistant_msg = ChatMessage(
                chat_id=chat.id,
                role=MessageRole.ASSISTANT,
                content=content,
                sequence_index=user_message.sequence_index + 1,
                is_in_qdrant=False,
                has_attachment=False
            )
            db.add(assistant_msg)
            chat.message_count += 2
            await db.commit()

            log.info("assistant_message_saved", chat_id=str(chat.id))

        except Exception:
            log.exception("failed_to_save_assistant_message", chat_id=str(chat.id))


llm_service = LLMService()