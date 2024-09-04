
# RAG-based Chat APIs using Google Gemini, Pinecone, and Firebase



This project provides a set of Flask-based APIs that enable users to create a conversational AI system powered by Retrieval-Augmented Generation (RAG). The system leverages Google Gemini, Pinecone, and Firebase to offer a highly responsive and intelligent query-answering experience.

Key Features:
Document Upload & Processing: Users can upload PDF documents, which are automatically split into manageable chunks and stored as vectors in a Pinecone vector database.

Retrieval-Augmented Generation (RAG): When a query is made, the system retrieves relevant document chunks from the Pinecone vector store and passes them to Google Gemini for generating accurate, context-aware responses.

Google Gemini Integration: Utilizes Google Gemini's powerful language model for embeddings and generating natural language responses based on the retrieved content.

Pinecone Vector Store: Efficient storage and retrieval of document chunks, enabling fast and accurate response generation.

Firebase Integration: Manages metadata and indexing information, linking document uploads with the corresponding vector indexes in Pinecone.


## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.7+**: [Download Python](https://www.python.org/downloads/)
- **pip**: Python package installer.
- **Virtual Environment** (Optional but recommended)

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/koti-malla/your-repo.git
cd your-repo

```

### 2. Create a Virtual Environment (Optional)
It's recommended to create a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  
# On Windows use 
\venv\Scripts\activate

```
### 3. Install Required Packages
Ensure you have a requirements.txt file in the project directory, then install the dependencies:
```bash
pip install -r requirements.txt

```
### 4. Add Credentials in .env File
Create a .env file in the project root and add the following environment variables:
```bash
FLASK_SECRET_KEY=your_flask_secret_key

FIREBASE_CRED_PATH=path_to_your_firebase_credential_json

PINECONE_API_KEY=your_pinecone_api_key

GOOGLE_API_KEY=your_google_api_key

```

### 5. Run the Flask Application
Start the Flask server by running:
```bash
python app.py

```
The server will start at http://127.0.0.1:5000.
## Test APIs

#### Testing the Upload Endpoint
You can test the **/upload**  endpoint using the following Python script. This script will upload a PDF file to the server:

```bash
 import requests


url = 'http://127.0.0.1:5000/upload'


file_path = r"C:\path\to\your\file.pdf"  # Replace with your file path
chat_name = 'XXXXXXXXXXX'  # Replace with your chat name


with open(file_path, 'rb') as file:
    
    files = {'file': (file_path, file, 'application/pdf')}
    data = {'chat_name': chat_name}

    response = requests.post(url, files=files, data=data)

    try:
        print(response.json())

    except ValueError:
        print("Response is not in JSON format:", response.text)

```

#### Expected Response 

Upon successful upload, you should see a response similar to:
```bash
{
    "message": "Document uploaded and processed"
}

```

#### Querying the Uploaded Document
To query the uploaded document, use the following Python script:

```bash
import requests

from IPython.display import Markdown


url = 'http://127.0.0.1:5000/query'

query = "What are the steps to run the file.py?" 

chat_name = 'text-analysis'  # Replace with your chat name

data = {"query": query, 'chat_name': chat_name}

response = requests.post(url, data=data)


try:
    display(Markdown(response.json()["response"]))

except ValueError:

    print("Response is not in JSON format:", response.text)

````

Expected Response:

The response should provide a detailed and accurate answer to your query based on the context extracted from the uploaded document.

## Demo

https://drive.google.com/file/d/1yRK4XbJFd9oQZvR0dmsauhJLC90wq9mf/view?usp=sharing
