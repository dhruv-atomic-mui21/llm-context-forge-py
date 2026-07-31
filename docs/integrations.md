# Framework Integrations

`llm-context-forge` provides first-class integrations for **LangChain** and **LlamaIndex**.

---

## LangChain Integration

```python
from llm_context_forge.integrations.langchain import ContextForgeTextSplitter, ContextForgeDocumentTransformer
from llm_context_forge.chunker import ChunkStrategy

# TextSplitter
splitter = ContextForgeTextSplitter(model="gpt-4o", strategy=ChunkStrategy.PARAGRAPH, max_tokens=500)
chunks = splitter.split_text("Your raw text here...")

# DocumentTransformer (Async supported)
transformer = ContextForgeDocumentTransformer(model="gpt-4o", max_tokens=500)
transformed_docs = await transformer.atransform_documents(langchain_docs)
```

---

## LlamaIndex Integration

```python
from llm_context_forge.integrations.llamaindex import ContextForgeNodeParser

parser = ContextForgeNodeParser(max_tokens=500)
nodes = parser.split_text_metadata_aware("def foo(): return 42", metadata_str="file_type: python code")
```
