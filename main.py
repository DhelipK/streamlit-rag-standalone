import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Load vector DB
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

# Streamlit UI
st.set_page_config(page_title="Banking Assistant RAG",page_icon="💼")
st.title("Internal Banking Assitant")

query = st.text_input("Ask a question about the Bank FD/RD:")

if query:
    # Retrieve relevant chunks
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Format prompt
    template = ChatPromptTemplate.from_template("""
    You are a helpful banking assistant. Use only the context provided.
    If unsure, say "I'm not sure based on the documents provided."

    Context:
    {context}

    Question:
    {question}
    """)

    prompt = template.format_messages(context=context,question=query)

    # Generate response
    response =  llm.invoke(prompt)

    # Display the result
    st.subheader("📌 Answer")
    st.markdown(response.content)

    with st.expander("Sourcs:"):
        for i, doc in enumerate(relevant_docs):
            st.markdown(f"**Source {i+1}:**\n{doc.page_content[:500]}...")



