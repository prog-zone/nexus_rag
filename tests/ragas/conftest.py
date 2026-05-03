import pytest
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@pytest.fixture(scope="session")
def ragas_llm():
    return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))


@pytest.fixture(scope="session")
def ragas_embeddings():
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings())

def pytest_addoption(parser):
    parser.addoption("--dataset", default="ragas_dataset.json")