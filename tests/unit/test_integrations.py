"""
Unit tests for LangChain and LlamaIndex Integrations.
"""

import pytest
from llm_context_forge.integrations.langchain import ContextForgeTextSplitter, ContextForgeDocumentTransformer
from llm_context_forge.integrations.llamaindex import ContextForgeNodeParser
from llm_context_forge.chunker import ChunkStrategy


def test_langchain_text_splitter():
    splitter = ContextForgeTextSplitter(model="gpt-4o", strategy=ChunkStrategy.PARAGRAPH, max_tokens=20)
    text = "Paragraph 1 is here.\n\nParagraph 2 is over here.\n\nParagraph 3 is also here."
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_langchain_document_transformer():
    transformer = ContextForgeDocumentTransformer(model="gpt-4o", max_tokens=20)

    class DummyDoc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {"source": "test"}

    docs = [DummyDoc("Sentence 1. Sentence 2. Sentence 3. Sentence 4.")]

    res = transformer.transform_documents(docs)
    assert len(res) >= 1

    async_res = await transformer.atransform_documents(docs)
    assert len(async_res) >= 1


def test_llamaindex_node_parser():
    parser = ContextForgeNodeParser(max_tokens=30)
    code_text = "def hello():\n    print('world')\n\ndef foo():\n    return 42"
    chunks = parser.split_text_metadata_aware(code_text, "file_type: python code")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
