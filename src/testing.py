from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import StorageContext,load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.schema import MetadataMode

import time


# Set the embeddings model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-base",
    # max_length=512
)

# Set up the LLAMA index
Settings.llm = Ollama(model="llama2", request_timeout=30.0)


persistent_dir = "vector_stores/multilingual_e5_base/"
storage_context = StorageContext.from_defaults(persist_dir=persistent_dir)


# Create a query engine
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
nodes = query_engine.retrieve("Que es un modelo de lente fina? explicamelo en detalle en 8 lineas")
for node in nodes:
    print(node.get_content(metadata_mode=MetadataMode.LLM))
    print(node.get_content(metadata_mode=MetadataMode.EMBED))
    print('-----------------HEY-----------------')

response = query_engine.query("Que es un modelo de lente fina? explicamelo en detalle en 8 lineas")
print(response.response)
# Perform a quer