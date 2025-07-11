from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="qwen3:1.7b", base_url="http://localhost:11434")

chain = prompt | model

print(chain.invoke({"question": "Tell me a dad joke"}))