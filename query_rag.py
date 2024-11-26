from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a virtual chat assistant for answering AWS customer and AWS service agreement questions. Answer the question based on the given context. If you don't know the answer, just say you don't know and don't make up an answer. The answer needs to be factual and based on the context given."),
    MessagesPlaceholder(variable_name="context"),
    ("user", "{question}")
])

#llm
llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature=0.2,
    api_key="",
    max_retries=2,
    max_tokens=None
)

# embedding = OpenAIEmbeddings(openai_api_key="") 

embedding = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-mistral-7b-v2", 
            api_key="", 
            truncate="NONE", 
        )

vectordb = Chroma(
    persist_directory='chroma',
    embedding_function=embedding
)

doc_count = vectordb._collection.count()
print(f"Vector store contains {doc_count} documents.")

retriever = vectordb.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = RunnableMap({
    "context": retriever,
    "question": RunnablePassthrough()
}) | prompt | llm | StrOutputParser()


query = "What is Amazon Fraud Detector terms mentioned in service terms?"

retrieved_docs = retriever.invoke(query)
print(f"Retrieved {len(retrieved_docs)} documents")

for doc in retrieved_docs:
    print(doc.metadata['source'])
    print(doc.page_content[:200])

result = rag_chain.invoke(query)

print("Answer:", result)