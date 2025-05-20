import streamlit as st
import openai
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.schema.document import Document

# ========= API Keys =========
openai.api_key = "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"  # You can load this from .env securely

# ========= Qdrant Connection =========
collection_name = "ncert_openai_embeddings"
client = QdrantClient(url="http://localhost:6333")
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)

vectorstore = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embedding_model,
)

# ========= Streamlit UI =========
st.title("📚 NCERT GPT-QA Chatbot")
query = st.text_input("❓ Enter your question here")

selected_class = st.selectbox("Select Class", [str(i) for i in range(1, 13)])
selected_subject = st.selectbox("Select Subject", [
    "english", "science", "math", "biology", "physics", "chemistry", "accountancy",
    "geography", "history", "computer_science", "home_science", "physical_education",
    "biotechnology", "business_studies", "economics", "psychology", "sociology",
    "political_science", "environmental_science", "vocational_education"
])

top_k = st.slider("🔍 Number of context chunks", 1, 10, 3)
temp = st.slider("🔥 Creativity (temperature)", 0.0, 1.0, 0.7)

if st.button("Get Answer") and query:
    with st.spinner("🔎 Searching & Generating..."):

        # ====== Perform Vector Similarity Search with Metadata Filters ======
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            filter={
                "class": selected_class,
                "subject": selected_subject.lower()
            }
        )

        # ====== Extract Context from Retrieved Chunks ======
        context = "\n\n".join([doc.page_content for doc, _ in results])

        # ====== Prompt Construction for GPT ======
        system_prompt = "You are a helpful educational assistant trained on NCERT books. Use the context to answer factually."
        user_prompt = f"""Context:
{context}

Question: {query}
Answer:"""

        # ====== OpenAI Chat Completion ======
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp
            )

            answer = response['choices'][0]['message']['content']
            st.markdown("### 🧠 Generated Answer")
            st.write(answer)

            with st.expander("📖 View Retrieved Context"):
                for doc, score in results:
                    st.markdown(f"**Score:** {score:.2f}")
                    st.text(doc.page_content)

        except Exception as e:
            st.error(f"❌ Failed to generate response: {e}")
