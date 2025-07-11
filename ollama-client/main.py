from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

# TODO understand how to tune OllamaLLM constructor parameters
# num_ctx, num_gpu, num_predict, num_thread, reasoning...
# Reference: https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html
model = OllamaLLM(model="qwen3:1.7b", base_url="http://localhost:11434")

chain = prompt | model

print(chain.invoke({"question": "Tell me a dad joke"}))