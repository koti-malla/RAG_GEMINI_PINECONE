from flask import Flask, request, jsonify
import os
import time
from uuid import uuid4
from werkzeug.utils import secure_filename
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import firebase_admin
from firebase_admin import credentials, db
from langchain_core.runnables import RunnablePassthrough 
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Secret key for session management
app.secret_key = os.getenv('SECRET_KEY')

# Folder for uploading files
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Firebase setup
cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS_PATH'))
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv('FIREBASE_DB_URL')
})
firebase_db = db.reference()


# Pinecone setup
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
ALLOWED_EXTENSIONS = {'pdf'}
# Embeddings setup
google_api_key = os.getenv('GOOGLE_API_KEY')
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, 
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, 
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, 
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=google_api_key,
    safety_settings=safety_settings
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'chat_name' not in request.form:
        return jsonify({'error': 'No file or chat name provided'}), 400

    file = request.files['file']
    chat_name = request.form['chat_name']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Ensure the upload directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call document processing and indexing logic
        try:
            return upload_document(file_path, chat_name)
        finally:
            # Clean up the saved file
            os.remove(file_path)
    else:
        return jsonify({'message': 'File type is not allowed, only PDFs are allowed'}), 400

def upload_document(file_path, chat_name):
    # Load and process the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    # Generate a unique index name based on the chat name
    index_name = f"{chat_name[:3]}-rag-gemini123"
    
    # Store chat_name and index_name in Firebase
    firebase_db.child(chat_name).set(index_name)

    # Check if the index already exists in Pinecone
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    # Store document chunks in Pinecone
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)

    return jsonify({'message': 'Document uploaded and indexed successfully'}), 200

@app.route('/query', methods=['POST'])
def query():
    if 'chat_name' not in request.form or 'query' not in request.form:
        return jsonify({'error': 'No chat name or query provided'}), 400

    chat_name = request.form['chat_name']
    query = request.form['query']

    # Fetch the index name from Firebase
    index_name = firebase_db.child(chat_name).get()
    if not index_name:
        return jsonify({'error': 'Index name not found for the given chat name'}), 404

    # Retrieve documents from Pinecone
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vector_store.as_retriever()

    template = """You are a conversational AI designed to assist users with information retrieval and question answering.

### Context:
{context}

### User Query:
{question}

### Instructions:
Given the context extracted from the document and the user's query, generate a detailed and accurate response.
Ensure that your answer is directly related to the context provided and addresses the user's question comprehensively. 
If the answer is not available in the context, politely inform the user that the information is not available.

### Response:
"""

    prompt = PromptTemplate(input_variables=["question", "context"], template=template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm # Replace with actual LLM
        | StrOutputParser()
    )

    try:
        result = rag_chain.invoke(query)
        return jsonify({'response': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
