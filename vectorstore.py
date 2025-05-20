import os
import zipfile
import tempfile
import backoff
import tiktoken
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from openai import RateLimitError

# ====== Initialize tiktoken encoding for ada embeddings ======
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

# ====== OpenAI Embeddings Setup ======
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key="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"
)
print("✅ OpenAI Embedding model loaded...")

# ====== Qdrant Setup ======
client = QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=False
)
collection_name = "ncert_openai_embeddings"

# ====== TokenTextSplitter with tiktoken ======
text_splitter = TokenTextSplitter(
    chunk_size=200,   # max tokens per chunk
    chunk_overlap=20,
    encoding_name="cl100k_base"  # encoding compatible with OpenAI ada model
)

# ====== Helper: Count tokens in text ======
def count_tokens(text: str) -> int:
    tokens = encoding.encode(text)
    return len(tokens)

# ====== PDF Processor ======
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
        # Optional: print token count per chunk for debugging
        tokens_in_chunk = count_tokens(doc.page_content)
        print(f"Chunk tokens: {tokens_in_chunk}")
    return chunks

# ====== Retry Decorator for RateLimitError ======
@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
def embed_batch(docs_batch):
    return embeddings.embed_documents([doc.page_content for doc in docs_batch])

# ====== Batch Embedding with Retry ======
def store_in_batches(all_docs, batch_size=10):
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        try:
            # Precompute embeddings separately
            embeddings_list = embed_batch(batch)
            Qdrant.from_embeddings(
                embeddings=embeddings_list,
                documents=batch,
                embedding=embeddings,
                client=client,
                collection_name=collection_name
            )
            print(f"✅ Stored batch {i//batch_size + 1}")
        except Exception as e:
            print(f"❌ Error storing batch {i//batch_size + 1}: {e}")

# ====== Main Folder Traversal ======
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

                subject_guess = os.path.splitext(zip_file)[0].lower()
                is_relevant = any(keyword.lower() in subject_guess for keyword in [
                    "english", "science", "env_science", "math", "biology",
                    "physics", "chemistry", "accountancy", "geography", "history",
                    "computer_science", "home_science", "physical_education", 
                    "biotechnology", "business_studies", "economics", "psychology",
                    "sociology", "political_science", "environmental_science", 
                    "vocational_education"
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
        store_in_batches(all_docs, batch_size=8)  # limit batch size
        print("✅ Embeddings stored in Qdrant.")
    else:
        print("⚠️ No documents were processed.")

# ====== Entry Point ======
if __name__ == "__main__":
    store_embeddings()



