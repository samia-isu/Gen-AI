import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool")
st.sidebar.title("News Article URls")

# urls = []

# for i in range(3):
#     url = st.sidebar.text.input(f"URL{i+1}")
#     urls.append(url)
#     #     st.write(f"Input for URL {i+1}: {urls}")
# for i in range(3):  # Example loop to create multiple input fields
#     urls = st.sidebar.text_input(f"URL{i+1}")
#     st.write(f"Input for URL {i+1}: {urls}")
num_urls = 3  # Number of URLs to input
urls = []

for i in range(num_urls):
    url = st.sidebar.text_input(f"Enter URL {i+1}")
    urls.append(url)
    
st.write("Entered URLs:")
for i, url in enumerate(urls, start=1):
    st.write(f"URL {i}: {url}")

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(model_name= "gpt-3.5-turbo", temperature = 0.9, max_tokens = 500)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    vectorstore_openai = FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)