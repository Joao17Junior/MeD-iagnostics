from pathlib import Path

import streamlit as st

from src.pipeline import RAG_Pipeline


RAW_DIR = Path("data/raw")
INPUT_DIR = Path("data/input_data")


@st.cache_resource
def get_pipeline() -> RAG_Pipeline:
    return RAG_Pipeline()


def save_uploads(uploaded_files, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    saved = 0
    for uploaded_file in uploaded_files or []:
        (destination / uploaded_file.name).write_bytes(uploaded_file.getbuffer())
        saved += 1
    return saved


st.set_page_config(page_title="MeD-iagnostics", page_icon="+", layout="centered")
st.title("MeD-iagnostics")

st.subheader("Add documents")
raw_files = st.file_uploader(
    "Files for the knowledge base",
    type=["pdf", "dcm", "dicom"],
    accept_multiple_files=True,
    key="raw_files",
)
if st.button("Save to data/raw"):
    count = save_uploads(raw_files, RAW_DIR)
    st.success(f"Saved {count} file(s) to {RAW_DIR}")

input_files = st.file_uploader(
    "Files for the current query",
    type=["pdf", "dcm", "dicom"],
    accept_multiple_files=True,
    key="input_files",
)
if st.button("Save to data/input_data"):
    count = save_uploads(input_files, INPUT_DIR)
    st.success(f"Saved {count} file(s) to {INPUT_DIR}")

st.divider()

pipeline = get_pipeline()

st.subheader("Knowledge base")
if st.button("Ingest data/raw"):
    try:
        count = pipeline.ingest(raw_dir=str(RAW_DIR))
        st.success(f"Ingested {count} chunk(s)")
    except Exception as error:
        st.error(f"Ingestion failed: {error}")

st.subheader("Ask a question")
query = st.text_area("Question", placeholder="Describe the information you need")
n_results = st.number_input("Results", min_value=1, max_value=20, value=5, step=1)

if st.button("Ask"):
    if not query.strip():
        st.warning("Enter a question first")
    else:
        try:
            with st.spinner("Searching and generating a response..."):
                response = pipeline.query(
                    query_text=query,
                    n_results=int(n_results),
                    input_folder=str(INPUT_DIR),
                )
            st.write(response)
        except Exception as error:
            st.error(f"Query failed: {error}")
