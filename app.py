import json
import streamlit as st
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_documents(resume):
    """Convert resume data into searchable documents with rich context"""
    docs = []

    # Basic Info
    basics = f"Name: {resume['basics']['name']}\nSummary: {resume['basics']['summary']}"
    docs.append(Document(
        page_content=basics,
        metadata={"section": "summary", "type": "overview"}
    ))

    # Employment History
    for company in resume['employment']:
        for position in company['positions']:
            for highlight in position['highlights']:
                content = f"{position['title']} at {company['company']}:\n"
                content += f"Achievement: {highlight['achievement']}\n"
                content += f"Technologies: {', '.join(highlight.get('technologies', []))}\n"

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "type": "experience",
                        "company": company['company'],
                        "role": position['title'],
                        "technologies": highlight.get('technologies', []),
                        **highlight.get('metrics', {})
                    }
                ))

    # Technical Skills
    for category in resume['technical_skills']['categories']:
        content = f"{category['name']} ({category['experience']} experience):\n"
        content += f"Used in {category['projects']} projects: {', '.join(category['items'])}"

        docs.append(Document(
            page_content=content,
            metadata={
                "type": "skills",
                "category": category['name'],
                "experience": category['experience'],
                "projects": category['projects']
            }
        ))

    # Projects
    for project in resume['projects']:
        content = f"Project: {project['name']}\nRole: {project['role']}\n"
        content += f"Technologies: {', '.join(project.get('technologies', []))}\n"
        content += "Achievements:\n- " + "\n- ".join(
            [f"{a['description']} ({a['impact']})" for a in project.get('achievements', [])]
        )

        docs.append(Document(
            page_content=content,
            metadata={
                "type": "project",
                "technologies": project.get('technologies', []),
                "duration": project.get('duration', "Unknown")
            }
        ))

    return docs

def load_resume():
    """Load and validate resume data"""
    try:
        with open("resum.json") as f:
            data = json.load(f)

        # Validate critical fields
        required_fields = ['basics', 'employment', 'technical_skills']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        return data
    except Exception as e:
        st.error(f"Error loading resume: {str(e)}")
        st.stop()

# Initialize application
def main():
    st.set_page_config(
        page_title="Resume Assistant",
        page_icon="💼",
        layout="wide"
    )

    # Load data
    resume_data = load_resume()
    documents = create_documents(resume_data)
    vector_store = FAISS.from_documents(documents, embeddings)

    # Sidebar Profile Section
    with st.sidebar:
        st.image("profile.jpg", width=150)  # Replace with your actual profile image path
        st.header("Quick Facts")
        display_quick_facts(resume_data)

        # Resume Download Button
        with open("resume.pdf", "rb") as resume_file:  # Replace with your actual resume file path
            resume_bytes = resume_file.read()
            st.download_button(
                label="📥 Download My Resume",
                data=resume_bytes,
                file_name="Sajith_Resume.pdf",
                mime="application/pdf"
            )

    # UI Components
    st.title("🔍 Resume Assistant")
    st.write("Ask about my experience, skills, or projects!")

    # Search Interface
    query = st.text_input("Enter your question:", 
                          placeholder="e.g. What AWS experience do you have?",
                          help="Ask about technologies, projects, or achievements")

    if query:
        results = vector_store.similarity_search(query, k=3)
        display_results(results)

def display_results(results):
    """Display search results with context-aware formatting"""
    primary_result = results[0]

    # Direct Answer Section
    with st.container():
        st.subheader("Most Relevant Answer")
        st.info(primary_result.page_content)

        # Show metadata context
        if 'technologies' in primary_result.metadata:
            st.write(f"**Technologies:** {', '.join(primary_result.metadata['technologies'])}")

        if 'company' in primary_result.metadata:
            st.write(f"**Experience Context:** {primary_result.metadata['company']}")

    # Detailed Context Section
    with st.expander("See additional relevant information"):
        for i, doc in enumerate(results[1:], 1):
            st.markdown(f"**Match {i}**")
            st.write(doc.page_content)
            st.divider()

def display_quick_facts(resume):
    """Show key metrics and contact info"""
    basics = resume['basics']

    st.write(f"**Name:** {basics['name']}")
    st.write(f"**Experience:** {resume['metrics']['total_experience']}")
    st.write(f"**Projects:** {resume['metrics']['projects_completed']}")
    st.write(f"**Location:** {basics['contact']['location']}")

    st.markdown("---")
    st.markdown("**Contact:**")
    st.write(f"📧 {basics['contact']['email']}")
    st.write(f"📱 {basics['contact']['phone']}")
    st.markdown(f"[LinkedIn Profile]({basics['contact']['linkedin']})")

    st.markdown("---")
    st.markdown("### Key Expertise 🛠️")
    st.markdown("- Java Spring Boot projects ☕\n"
                "- Team leadership 👥\n"
                "- AWS Cloud experience ☁️\n"
                "- Recent technologies 🛠️\n"
                "- Database expertise: **MySQL, Elasticsearch, Neo4J** 🗄️")

if __name__ == "__main__":
    main()
