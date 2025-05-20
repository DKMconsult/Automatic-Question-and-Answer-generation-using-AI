import os
import zipfile
import tempfile
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient

# ====== OpenAI Embeddings Setup ======
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # Most efficient embedding model from OpenAI
    openai_api_key="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"  # Or set via environment variable
)
print("✅ OpenAI Embedding model loaded...")

# ====== Qdrant Setup (local or cloud) ======
client = QdrantClient(
    url="http://localhost:6333",  # Or use Qdrant cloud URL
    prefer_grpc=False
)
collection_name = "ncert_openai_embeddings"

# ====== Text Splitter ======
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# ====== PDF Processing Function ======
def process_pdf(pdf_path, class_name, subject, chapter_name):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    for doc in chunks:
        doc.metadata = {
            "class": class_name,
            "subject": subject,
            "chapter": chapter_name
        }
    return chunks

# ====== Full Folder Traversal Function ======
def store_embeddings(base_path=r"D:\MLOps\Raspect\LoveMyTest_Project\1to10"):
    all_docs = []

    for class_folder in os.listdir(base_path):
        class_path = os.path.join(base_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        for zip_file in os.listdir(class_path):
            if not zip_file.endswith(".zip"):
                continue

            zip_path = os.path.join(class_path, zip_file)

            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                except Exception as e:
                    print(f"❌ Failed to unzip {zip_file}: {e}")
                    continue

                # Extract subject name heuristically from filenames or folder structure
                subject_guess = os.path.splitext(zip_file)[0].lower()
                is_relevant = any(keyword in subject_guess for keyword in [
                    "english", "science","Env_Science","Vocational_Education","Environmental_Science","math", "biology", "physics", "chemistry","accountancy","Geography","History","Computer_Science","Home_Science","Physical_Education","Biotechnology","Business_Studies","Economics","Psychology","Sociology","Political_Science"
                ])

                if not is_relevant:
                    print(f"⏭️ Skipping irrelevant subject zip: {zip_file}")
                    continue

                for root, _, files in os.walk(tmpdir):
                    for pdf_file in files:
                        if not pdf_file.endswith(".pdf"):
                            continue

                        chapter_name = os.path.splitext(pdf_file)[0]
                        pdf_path = os.path.join(root, pdf_file)

                        print(f"📄 Processing: {pdf_path}")
                        try:
                            chunks = process_pdf(pdf_path, class_folder, subject_guess, chapter_name)
                            all_docs.extend(chunks)
                        except Exception as e:
                            print(f"❌ Error processing {pdf_path}: {e}")

    print(f"🔢 Total chunks processed: {len(all_docs)}")

    if all_docs:
        Qdrant.from_documents(
            documents=all_docs,
            embedding=embeddings,
            client=client,
            collection_name=collection_name
        )
        print("✅ Embeddings stored in Qdrant using OpenAI.")
    else:
        print("⚠️ No documents were processed.")

# ====== Entry Point ======
if __name__ == "__main__":
    store_embeddings()