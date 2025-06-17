from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document,StorageContext, ServiceContext, set_global_handler, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import qdrant_client

from llama_index.vector_stores.qdrant import QdrantVectorStore

import datetime, uuid
from llama_index.core.schema import TextNode
from llama_index.core.postprocessor.node_recency import FixedRecencyPostprocessor


# Ollama LLM과 임베딩 모델 설정
Settings.llm = Ollama(model="gemma3:latest", request_timeout=60.0, system_prompt="You are a helpful assistant that answers like a knowledge base agent.")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")

# turn on debugging
set_global_handler("simple")

client = qdrant_client.QdrantClient(
    path="./qdrant_data"
)
vector_store = QdrantVectorStore(client=client, collection_name="slack_messages")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# index = VectorStoreIndex([],storage_context=storage_context)

index = VectorStoreIndex([], storage_context=storage_context)



dt_object = datetime.datetime.now()
formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')

template = (
    "Your context is a series of chat messages. Each one is tagged with 'who:' \n"
    "indicating who was speaking and 'when:' indicating when they said it, \n"
    "followed by a line break and then what they said. There can be up to 20 chat messages.\n"
    "The messages are sorted by recency, so the most recent one is first in the list.\n"
    "The most recent messages should take precedence over older ones.\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "You are a helpful AI assistant who has been listening to everything everyone has been saying. \n"
    "Given the most relevant chat messages above, please answer this question: {query_str}\n"
)
qa_template = PromptTemplate(template)                



# doc1 = Document(text="Bbomi is a cat")
# doc2 = Document(text="Doug is a dog")
# doc3 = Document(text="Carl is a rat")

# index.insert(doc1)
# index.insert(doc2)
# index.insert(doc3)

for text in ["Bbomi is a cat", "Doug is a dog", "Carl is a rat","Bbomi is a dog"]:
    # create a node with metadata
    node = TextNode(
        text=text,
        id_=str(uuid.uuid4()),
        metadata={
            "when": formatted_time
        }
    )
    index.insert_nodes([node])


# run a query

postprocessor = FixedRecencyPostprocessor(
    top_k=20, 
    date_key="when", # the key in the metadata to find the date

)

query_engine = index.as_query_engine(similarity_top_k=20, node_postprocessors=[postprocessor])

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_template}
)

response = query_engine.query("Who is Bbomi?")
print(response)


