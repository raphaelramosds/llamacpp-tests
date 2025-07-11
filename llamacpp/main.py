# import bs4
import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
# from langchain_community.embeddings import LlamaCppEmbeddings
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.documents import Document
# from langgraph.graph import START, StateGraph
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from typing_extensions import List, TypedDict

local_model = "./models/llama-2-7b-chat.Q2_K.gguf"

llm = ChatLlamaCpp(
    temperature=0.2,
    model_path=local_model,
    n_ctx=4096,
    n_gpu_layers=8,
    n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    max_tokens=512,
    n_threads=multiprocessing.cpu_count() - 1,
    repeat_penalty=1.5,
    top_p=0.5,
    # verbose=True,
)

# embeddings = LlamaCppEmbeddings(model_path=local_model)

# vector_store = InMemoryVectorStore(embeddings)

# # Load and chunk contents of the blog
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# all_splits = text_splitter.split_documents(docs)

# # Index chunks
# _ = vector_store.add_documents(documents=all_splits)

# # Define prompt for question-answering
# # N.B. for non-US LangSmith endpoints, you may need to specify
# # api_url="https://api.smith.langchain.com" in hub.pull.
# # prompt = hub.pull("rlm/rag-prompt")

# prompt_structure = """
# Use the following context to answer the question.
# If you don't know the answer, just say you don't know.

# Context:
# {context}

# Question:
# {question}

# Answer:

# """

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             prompt_structure
#         ),
#         # Dynamically fill user messages on the placeholder below, allowing context-aware responses.
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )


# # Define state for application
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str


# # Define application steps
# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}


# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt.invoke({"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}


# # Compile application and test
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()

# response = graph.invoke({"question": "What is Task Decomposition?"})
# print(response["answer"])

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)