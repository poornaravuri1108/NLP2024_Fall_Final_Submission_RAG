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
import re
import csv
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
    elif embedding_model=="all-minilm":
        embedding = OllamaEmbeddings(
            model="all-minilm"
        )
    else:
        embedding = None
    return embedding

def create_vectorstore(doc_splits, embedding_chosen):
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
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

def extract_json(response_content):
    try:
        json_str = re.search(r'\{.*\}', response_content, re.DOTALL).group()
        response_json = json.loads(json_str)
        return response_json
    except Exception as e:
        return {'error': f'JSON extraction failed: {str(e)}', 'content': response_content}

def grade_hallucination_new(facts, generation, grading_llms):
    hallucination_grader_instructions = """
            You are a teacher grading a quiz.

            You will be given FACTS and a STUDENT ANSWER.

            Here is the grade criteria to follow:

            (1) Ensure the STUDENT ANSWER is grounded in the FACTS.

            (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

            Score:

            Provide a numeric score between 0 and 1 to indicate the correctness of the STUDENT ANSWER.

            - A score of 1 means that the student's answer meets all of the criteria and is perfectly grounded in the facts.

            - A score of 0 means that the student's answer does not meet any of the criteria and is completely incorrect.

            - Scores in between indicate partial correctness.

            Return the answer in the following JSON format without any additional text:

            {{
                "score": <numeric_score_between_0_and_1>,
                "explanation": "<your_explanation_here>"
            }}

            Do not include any other text outside the JSON output.

            Avoid simply stating the correct answer at the outset."""

    hallucination_grader_prompt = """
            FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}.

            Provide the JSON output as specified in the instructions.
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

        response_json = extract_json(result_content)

        responses.append({model_name: response_json})

    return responses

def main():
    load_dotenv()
    llms = ["gpt-3.5-turbo", "gemini-1.5-pro", "llama3.2:3b-instruct-fp16", "mistral", "llama2:latest", "gemma:2b"]
    grading_models = ["gpt-3.5-turbo"]
    os.environ["USER_AGENT"] = "NLP_Project/1.0"
    os.environ["LANGCHAIN_TRACING_V2"]="true"
    os.environ["LANGCHAIN_PROJECT"]="NLP_Project_rag"
    st.title("RAG Model with Multiple LLMs")
    st.write("Enter your question, select an embedding model, and select a model to generate a response.")
    embedding_model = st.selectbox("Select text embedding model", ["mxbai-embed-large", "text-embedding-3-large", "text-embedding-3-small", "nomic-embed-text-v1", "nomic-embed-text", "paraphrase-multilingual", "all-minilm"])
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
        st.write("### Grading Result:")
        for result in grading_result:
            for grading_model_name, grading in result.items():
                st.write(f"**Grading Model: {grading_model_name}**")
                if 'error' in grading:
                    st.write(f"Error: {grading['error']}")
                    st.write(f"Response Content: {grading['content']}")
                else:
                    # Check if 'score' and 'explanation' are in grading
                    score = grading.get('score', None)
                    explanation = grading.get('explanation', None)
                    if score is not None and explanation is not None:
                        st.write(f"Score: {score}")
                        st.write(f"Explanation: {explanation}")
                    else:
                        st.write(f"Incomplete grading response: {grading}")
        # Save results in JSONL format
        data_to_save = {
            'question': question,
            'embedding_model': embedding_model,
            'llm_model': model_name,
            'answer': answer,
            'grading_results': grading_result
        }
        with open('results.jsonl', 'a') as f:
            json.dump(data_to_save, f)
            f.write('\n')
        # Save results in CSV format
        csv_file_exists = os.path.isfile('results.csv')
        with open('results.csv', 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question', 'embedding_model', 'llm_model', 'answer', 'grading_model', 'score', 'explanation']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not csv_file_exists:
                writer.writeheader()
            for result in grading_result:
                for grading_model_name, grading in result.items():
                    row = {
                        'question': question,
                        'embedding_model': embedding_model,
                        'llm_model': model_name,
                        'answer': answer,
                        'grading_model': grading_model_name,
                        'score': grading.get('score', ''),
                        'explanation': grading.get('explanation', ''),
                    }
                    writer.writerow(row)

if __name__ == "__main__":
    main()
