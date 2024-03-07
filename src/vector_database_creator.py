from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import time

documents = SimpleDirectoryReader("test_data_folder/uni_data", recursive=True).load_data()

print(*documents, sep="\n")
# Set the embeddings model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-base",
    max_length=512
)


# print(Settings.embed_model.get_text_embedding("What did the author do growing up?"))
# Build the index
index = VectorStoreIndex.from_documents(documents,show_progress=True)
index.storage_context.persist("vector_stores/multilingual_e5_base")

