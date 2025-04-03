from flask import Blueprint, jsonify, Response, request
import logging
import os
import uuid
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from utils.config import pinecone_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chat_routes = Blueprint('upload', __name__)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_config.api_key)
index = pc.Index("finance")


def process_file_based_on_mime(file_path, metadata, doc_name):
    if file_path.lower().endswith('.pdf'):
        upload_pdf(metadata, doc_name, os.path.dirname(file_path))
    else:
        logger.warning(f"Unsupported file type: {file_path}")


def upload_pdf(meta_data, doc_name, directory_path):
    try:
        # use the directory path for DirectoryLoader
        loader = PyPDFDirectoryLoader(directory_path, glob="*.pdf")
        docs = loader.load()
        print("docs", docs)
        upload_documents(docs, meta_data, doc_name)
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise


def upload_documents(docs, meta_data, doc_name):
    try:
        for doc in docs:
            filename_with_extension = os.path.basename(doc.metadata["source"])
            docName, _ = os.path.splitext(filename_with_extension)
            text = doc.page_content

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=768)
            chunks = text_splitter.create_documents([text])

            embeddings_model = OpenAIEmbeddings()
            embeddings_arrays = embeddings_model.embed_documents(
                [chunk.page_content.replace("\n", " ") for chunk in chunks]
            )

            batch_size = 100
            batch = []

            for idx in range(len(chunks)):
                chunk = chunks[idx]
                metadata = {
                    "text": chunk.page_content,
                    "type": meta_data.strip('\"'),
                    "doc_link": str(doc_name)
                }
                vector = {
                    "id": str(uuid.uuid4()),
                    "values": embeddings_arrays[idx],
                    "metadata": metadata,
                }
                batch.append(vector)

                if len(batch) == batch_size or idx == len(chunks) - 1:
                    index.upsert(vectors=batch)
                    batch = []
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise


@chat_routes.route('/upload', methods=['POST'])
async def upload() -> tuple[Response, int] | Response:
    try:
        if request.method != 'POST':
            return jsonify(error='Method not allowed'), 405

        uploaded_files = request.files.getlist('files')
        metadata = request.form.get('type')

        if not metadata:
            return jsonify(error='Type required!'), 400
        if not uploaded_files:
            return jsonify(error='No files part in the request'), 400

        files_path = []
        try:
            # Ensure the upload directory exists
            upload_folder = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(upload_folder, exist_ok=True)

            for f in uploaded_files:
                # Define the path for each file
                upload_path = os.path.join(upload_folder, f.filename)
                # Save the file
                f.save(upload_path)

                files_path.append({"path": upload_path, "name": f.filename})
                process_file_based_on_mime(upload_path, metadata, f.filename)
                os.remove(upload_path)

            return jsonify(message='Files uploaded and processed successfully.')

        except Exception as e:
            # Clean up any files that were saved before the error occurred
            for file in files_path:
                if os.path.exists(file["path"]):
                    os.remove(file["path"])
            raise e

    except Exception as e:
        # Log error
        logger.error(f"Error processing upload request: {str(e)}")
        return jsonify({
            'error': f'Error processing request: {str(e)}'
        }), 500
