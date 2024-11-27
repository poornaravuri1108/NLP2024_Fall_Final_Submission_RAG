import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic import NomicEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import streamlit as st

def load_documents(urls):
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(docs)
    return doc_splits

def getEmbeddings(embedding_model):
    if embedding_model=="mxbai-embed-large":
        embedding = OllamaEmbeddings(
            model=embedding_model
        )

    elif embedding_model=="text-embedding-3-large":
        embedding = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.environ["OPENAI_API_KEY"]
        )

    elif embedding_model=="text-embedding-3-small":
        embedding = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.environ["OPENAI_API_KEY"]
        )

    elif embedding_model=="nomic-embed-text-v1":
        embedding = NomicEmbeddings(
            model=embedding_model, 
            nomic_api_key=os.environ["NOMIC_EMBED_API_KEY"]
        )

    elif embedding_model=="nomic-embed-text":
        embedding = OllamaEmbeddings(
            model="nomic-embed-text"
        )

    elif embedding_model=="paraphrase-multilingual":
        embedding = OllamaEmbeddings(
            model="paraphrase-multilingual"
        )

    else:
        embedding = None
    
    return embedding

def create_vectorstore(doc_splits, embedding_chosen):
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        # embedding=NomicEmbeddings(model="nomic-embed-text-v1", nomic_api_key=os.environ["NOMIC_EMBED_API_KEY"])
        embedding=embedding_chosen
    )
    return vectorstore

def retrieve_documents(question, retriever, k=3):
    docs = retriever.invoke(question)
    return docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_llm(model_name):
    if model_name == "gpt-3.5-turbo":
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0, api_key=os.environ["OPENAI_API_KEY"])
    elif model_name == "gemini-1.5-pro":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.2,
            api_key=os.environ["GEMINI_API_KEY"],
            max_tokens=None,
            max_retries=2
        )
    elif model_name == "llama3.2:3b-instruct-fp16":
        llm = OllamaLLM(
            model="llama3.2:3b-instruct-fp16",
            temperature=0.2,
        )
    elif model_name == "mistral":
        llm = OllamaLLM(
            model="mistral",
            temperature=0.2
        )
    elif model_name == "llama2:latest":
        llm = OllamaLLM(
            model="llama2:latest",
            temperature=0.2
        )
    elif model_name == "gemma:2b":
        llm = OllamaLLM(
            model = "gemma:2b",
            temperature=0.2
        )
    else:
        llm = None
    return llm

def generate_answer(question, context, model_name):
    rag_prompt = """You are an assistant for question-answering tasks. 
    
                    Here is the context to use to answer the question:

                    {context} 

                    Think carefully about the above context. 

                    Now, review the user question:

                    {question}

                    Provide an answer to this questions using only the above context. 

                    Use three sentences maximum and keep the answer concise.

                    Answer:"""
    
    rag_prompt_formatted = rag_prompt.format(context=context, question=question)
    llm = get_llm(model_name)
    if llm is None:
        return "Invalid model selected."
    if isinstance(llm, ChatOpenAI) or isinstance(llm, ChatGoogleGenerativeAI):
        messages = [HumanMessage(content=rag_prompt_formatted)]
        generation = llm.invoke(messages)
        return generation.content
    elif isinstance(llm, OllamaLLM):
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an assistant for question-answering tasks."),
            HumanMessage(content=rag_prompt_formatted)
        ])
        chain = prompt_template | llm | StrOutputParser()
        res = chain.invoke({})
        cleaned_output = res.replace("*", "").replace("•", "").strip()
        return cleaned_output
    else:
        return "Model not supported."

def grade_hallucination(facts, generation):
    hallucination_grader_instructions = """
            You are a teacher grading a quiz. 

            You will be given FACTS and a STUDENT ANSWER. 

            Here is the grade criteria to follow:

            (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

            (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

            Score:

            A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

            A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

            Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

            Avoid simply stating the correct answer at the outset."""
            
    hallucination_grader_prompt = """
            FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

            Return JSON with two keys: 'binary_score' is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS, and 'explanation' that contains an explanation of the score.
            """
    
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(documents=facts, generation=generation)
    llm = get_llm("gpt-3.5-turbo")
    messages = [SystemMessage(content=hallucination_grader_instructions), HumanMessage(content=hallucination_grader_prompt_formatted)]
    result = llm.invoke(messages)
    return json.loads(result.content)

def grade_hallucination_new(facts, generation, grading_llms):
    hallucination_grader_instructions = """
            You are a teacher grading a quiz. 

            You will be given FACTS and a STUDENT ANSWER. 

            Here is the grade criteria to follow:

            (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

            (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

            Score:

            A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

            A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

            Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

            Avoid simply stating the correct answer at the outset."""
            
    hallucination_grader_prompt = """
            FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

            Return JSON with two keys: 'binary_score' is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS, and 'explanation' that contains an explanation of the score.
            """
    
    responses = []
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(documents=facts, generation=generation)
    for model_name in grading_llms:
        llm = get_llm(model_name)
        messages = [SystemMessage(content=hallucination_grader_instructions), HumanMessage(content=hallucination_grader_prompt_formatted)]
        result = llm.invoke(messages)
        
        if hasattr(result, 'content'):
            result_content = result.content
        else:
            result_content = result
        
        try:
            response_json = json.loads(result_content)
        except json.JSONDecodeError as e:
            response_json = {'error': f'JSONDecodeError: {str(e)}', 'content': result_content}
        except Exception as e:
            response_json = {'error': f'Unexpected error: {str(e)}', 'content': result_content}
        
        responses.append({model_name: response_json})
    
    return responses

def main():
    load_dotenv()
    llms = ["gpt-3.5-turbo", "gemini-1.5-pro", "llama3.2:3b-instruct-fp16", "mistral", "llama2:latest", "gemma:2b"]
    grading_models = ["gpt-3.5-turbo", "mistral"]
    os.environ["USER_AGENT"] = "NLP_Project/1.0"
    os.environ["LANGCHAIN_TRACING_V2"]="true"
    os.environ["LANGCHAIN_PROJECT"]="NLP_Project_rag"
    st.title("RAG Model with Multiple LLMs")
    st.write("Enter your question, select an embedding model, and select a model to generate a response.")
    embedding_model = st.selectbox("Select text embedding model", ["mxbai-embed-large", "text-embedding-3-large", "text-embedding-3-small", "nomic-embed-text-v1", "nomic-embed-text", "paraphrase-multilingual"])
    model_name = st.selectbox("Select a model", llms)
    question = st.text_area("Enter your question")
    if st.button("Generate Answer"):
        urls = [
            "https://aws.amazon.com/agreement/",
            "https://aws.amazon.com/service-terms/"
        ]
        with st.spinner("Loading documents..."):
            docs = load_documents(urls)
        with st.spinner("Splitting documents..."):
            doc_splits = split_documents(docs)
        with st.spinner("Creating vectorstore..."):
            embedding = getEmbeddings(embedding_model)
            vectorstore = create_vectorstore(doc_splits, embedding)
            retriever = vectorstore.as_retriever(k=3)
        with st.spinner("Retrieving relevant documents..."):
            retrieved_docs = retrieve_documents(question, retriever)
            context = format_docs(retrieved_docs)
        with st.spinner("Generating answer..."):
            answer = generate_answer(question, context, model_name)
        st.write("### Answer:")
        st.write(answer)
        with st.spinner("Grading answer..."):
            grading_result = grade_hallucination_new(context, answer, grading_models)
        # st.write("### Grading Result:")
        # for result in grading_result:
        #     for model_name, response_content in result.items():
        #         st.write(f"**Model: {model_name}**")
        #         st.code(response_content, language='json')
        st.write("### Grading Result:")
        for result in grading_result:
            for model_name, grading in result.items():
                st.write(f"**Model: {model_name}**")
                if 'error' in grading:
                    st.write(f"Error: {grading['error']}")
                    st.write(f"Response Content: {grading['content']}")
                else:
                    st.write(f"Binary Score: {grading['binary_score']}")
                    st.write(f"Explanation: {grading['explanation']}")
        # with st.spinner("Grading answer..."):
        #     grading_result = grade_hallucination(context, answer)
        # st.write("### Grading Result:")
        # st.json(grading_result)

if __name__ == "__main__":
    main()
