from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    api_key=os.environ["GEMINI_API_KEY"],
    max_tokens=None,
    max_retries=2
)

prompt = ChatPromptTemplate([
    ("system", "You are a text generator based on the given context by human"),
    ("human", "Talk about {input_context}")
])

chain = prompt | llm

text_generated = chain.invoke(
    {
        "input_context": "Generative AI"
    }
)

print(text_generated.content)