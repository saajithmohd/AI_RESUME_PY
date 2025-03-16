import json
import numpy as np
import faiss
import streamlit as st
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_documents(resume):
    """Convert resume data into searchable documents"""
    docs = []

    # Basic Info
    basics = f"Name: {resume['basics']['name']}\nSummary: {resume['basics']['summary']}"
    docs.append(Document(page_content=basics, metadata={"section": "summary"}))

    # Employment History
    for company in resume['employment']:
        for position in company['positions']:
            for highlight in position['highlights']:
                content = f"{position['title']} at {company['company']}:\n"
                content += f"Achievement: {highlight['achievement']}\n"
                content += f"Technologies: {', '.join(highlight.get('technologies', []))}\n"

                docs.append(Document(
                    page_content=content,
                    metadata={"type": "experience", "company": company['company']}
                ))

    # Technical Skills
    for category in resume['technical_skills']['categories']:
        content = f"{category['name']} ({category['experience']}): {', '.join(category['items'])}"
        docs.append(Document(page_content=content, metadata={"type": "skills"}))

    # Projects
    for project in resume['projects']:
        content = f"Project: {project['name']}\nRole: {project['role']}\n"
        content += f"Technologies: {', '.join(project.get('technologies', []))}\n"
        docs.append(Document(page_content=content, metadata={"type": "project"}))

    return docs

def load_resume():
    """Load and validate resume data"""
    try:
        with open("resum.json") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading resume: {str(e)}")
        st.stop()

def initialize_faiss(documents):
    """Create FAISS vector index from resume documents"""
    texts = [doc.page_content for doc in documents]
    embeddings = np.array(embedding_model.embed_documents(texts))
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    return faiss_index, documents


def main():
    st.set_page_config(page_title="Resume Assistant", page_icon="💼", layout="wide")

    # Mobile-first CSS for better visibility
    st.markdown("""
    <style>
        @media (max-width: 768px) {
            .stDownloadButton > button {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 999;
                width: 90%;
                max-width: 300px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Load resume data and initialize FAISS
    resume_data = load_resume()
    documents = create_documents(resume_data)
    faiss_index, doc_list = initialize_faiss(documents)

    # Sidebar Profile
    with st.sidebar:
        st.image("profile.jpg", width=150)
        st.header("Quick Facts")
        st.write(f"**Name:** {resume_data['basics']['name']}")
        st.write(f"**Experience:** {resume_data['metrics']['total_experience']} years")
        st.write(f"**Projects:** {resume_data['metrics']['projects_completed']}")
        st.write(f"**Location:** {resume_data['basics']['contact']['location']}")

        with open("resume.pdf", "rb") as resume_file:
            st.download_button(
                label="📥 Download Resume",
                data=resume_file.read(),
                file_name="Resume.pdf",
                mime="application/pdf",
                key="download-main"
            )

    # Main Content Area
    st.title("🔍 Resume Assistant")
    st.write("Ask about my experience, skills, or projects!")

    query = st.text_input("Enter your question:", 
                         placeholder="e.g. What Java experience do you have?",
                         help="Ask about technologies, projects, or achievements")

    # Handle "download resume" queries
    if query and any(word in query.lower() for word in ["resume", "download resume", "cv"]):
        with open("resume.pdf", "rb") as resume_file:
            st.download_button(
                label="📥 Click to Download Resume",
                data=resume_file.read(),
                file_name="Resume.pdf",
                mime="application/pdf",
                key="download-query"
            )

    elif query:
        # Convert query into embedding and search FAISS index
        query_embedding = np.array([embedding_model.embed_query(query)])
        D, I = faiss_index.search(query_embedding, k=3)  # Get top 3 results

        results = [doc_list[i] for i in I[0] if i != -1]
        display_results(results)

def display_results(results):
    """Show AI-powered search results"""
    if not results:
        st.warning("No relevant information found.")
        return

    st.subheader("Most Relevant Answer")
    st.info(results[0].page_content)

    # Additional context
    with st.expander("📖 See More Results"):
        for i, doc in enumerate(results[1:], 1):
            st.markdown(f"**Related Match {i}**")
            st.write(doc.page_content)
            st.divider()

if __name__ == "__main__":
    main()
