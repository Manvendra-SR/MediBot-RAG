from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import *
import os
app = Flask(__name__)

load_dotenv()

HUGGINGFACEHUB_API_TOKEN=os.environ.get('HUGGINGFACEHUB_API_TOKEN')
MISTRAL_API_KEY=os.environ.get('MISTRAL_API_KEY')

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

embeddings = download_hugging_face_embeddings()

vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.3,
    max_retries=2,
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke(msg)
    print("Response : ", response.content)
    return str(response.content)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 5000, debug= True)
