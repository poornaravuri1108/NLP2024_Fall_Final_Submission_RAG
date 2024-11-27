from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(
    model="llama3.2:3b-instruct-fp16",
    temperature=0.2,
    
)

prompt =  ChatPromptTemplate([
    ("system", "You are a good AI model in generating text based on the input given by the user"),
    ("human", "Generate text on {input_context}")
])

chain = prompt | llm | StrOutputParser()

res = chain.invoke({
    "input_context":"llm hallucination"
})

cleaned_output = res.replace("*", "").replace("•", "").strip()

print(cleaned_output)