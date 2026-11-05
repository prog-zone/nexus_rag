import pytest

# The RAGAS extras pull a langchain stack that breaks on some version
# combinations (ragas 0.4 + langchain-community 0.4 dropped ChatVertexAI, which
# ragas still imports). Collection used to abort the whole suite with
# ``Interrupted: 1 error``; now the evaluation tests skip themselves instead.
RAGAS_IMPORT_ERROR = None
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception as exc:  # pragma: no cover - depends on the installed extras
    RAGAS_IMPORT_ERROR = exc


@pytest.fixture(scope="session")
def ragas_llm():
    if RAGAS_IMPORT_ERROR is not None:
        pytest.skip(f"RAGAS extras unavailable: {RAGAS_IMPORT_ERROR}")
    return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))


@pytest.fixture(scope="session")
def ragas_embeddings():
    if RAGAS_IMPORT_ERROR is not None:
        pytest.skip(f"RAGAS extras unavailable: {RAGAS_IMPORT_ERROR}")
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings())


def pytest_addoption(parser):
    parser.addoption("--dataset", default="ragas_dataset.json")
