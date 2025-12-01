import streamlit as st
import base64
from pathlib import Path

from langchain.document_loaders import SeleniumURLLoader
#from langchain_community.document_loaders import SeleniumURLLoader
#from langchain_community.document_loaders import RequestsURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

class AnyQuestion:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="llama3")
        self.llm = Ollama(model="llama3", temperature=0.7, num_predict=512)

    # def set_background(self):
    #     """Set background image for the Streamlit page"""
    #     image_path = r""
        
    #     if not Path(image_path).exists():
    #         st.error(f"Image not found at: {image_path}")
    #         return
        
    #     with open(image_path, "rb") as img_file:
    #         img_base64 = base64.b64encode(img_file.read()).decode()
        
    #     page_bg_img = f"""
    #     <style>
    #     [data-testid="stAppViewContainer"] {{
    #         background-image: url("data:image/png;base64,{img_base64}");
    #         background-size: cover;
    #         background-position: center;
    #         background-attachment: fixed;
    #     }}
    #     [data-testid="stSidebar"] {{
    #         background-color: rgba(255, 255, 255, 0.9);
    #     }}
    #     </style>
    #     """
    #     st.markdown(page_bg_img, unsafe_allow_html=True)

    def load_and_process_url(self, url):
        try:
            loader = SeleniumURLLoader(urls=[url], driver=driver)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content found at the URL")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            vectorstore = InMemoryVectorStore.from_documents(texts, self.embeddings)
            return vectorstore
        except Exception as e:
            raise Exception(f"Error loading URL: {str(e)}")

    def generate_answer(self, vectorstore, question):
        try:
            prompt_template = """You are a helpful assistant. Use the following context to answer the question.

            Context: {context}

            Question: {question}

            Answer:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            docs = vectorstore.similarity_search(question, k=4)
            context = "\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.format(context=context, question=question)
            answer = self.llm.invoke(final_prompt)
            return answer
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")

    def run(self):
        #self.set_background()
        
        st.title("Electronics Bot - Ask Questions from Any URL")

        url = st.text_input("Enter the URL:")
        question = st.text_input("Enter your question:")

        if st.button("Get Answer"):
            if url and question:
                with st.spinner("Processing..."):
                    try:
                        vectorstore = self.load_and_process_url(url)
                        answer = self.generate_answer(vectorstore, question)
                        st.success("Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Please enter both a URL and a question.")

if __name__ == "__main__":
    app = AnyQuestion()
    app.run()
