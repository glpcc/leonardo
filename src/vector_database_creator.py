from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import time

documents = SimpleDirectoryReader("test_data_folder/uni_data", recursive=True).load_data()

good_documents = []
for i in documents:
    if i.get_text() != "":
        i.excluded_embed_metadata_keys = [
            "file_name",
            "file_type",
            "file_size",
            "creation_date",
            "last_modified_date",
            "last_accessed_date",
        ]
        i.excluded_llm_metadata_keys = [
            "file_name",
            "file_size",
            "file_type",
            "creation_date",
            "last_modified_date",
            "last_accessed_date",
        ]
        good_documents.append(i)
    else:
        print("Empty document found")

# for i in documents:
#     print(i.to_embedchain_format())

# Set the embeddings model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-base",
    max_length=512
)


# Build the index
index = VectorStoreIndex.from_documents(good_documents,show_progress=True)
index.storage_context.persist("vector_stores/multilingual_e5_base")

