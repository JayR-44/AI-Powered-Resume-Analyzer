"""Run with: streamlit run streamlit_app.py."""

import hashlib

import streamlit as st

from resume_checker.qa import answer_question, create_llm, groq_error_message
from resume_checker.text_splitter import split_pages
from resume_checker.uploads import load_uploads
from resume_checker.vector_store import build_vector_store


@st.cache_resource(scope="session", max_entries=1, ttl=3600, show_spinner=False)
def prepare_cvs(files):
    pages = load_uploads(files)
    chunks = split_pages(pages)
    return build_vector_store(chunks), len(pages), len(chunks)


def show_sources(passages):
    with st.expander("View supporting passages"):
        for number, passage in enumerate(passages, start=1):
            st.caption(
                f"Passage {number} · {passage.metadata['display_source']} · "
                f"Page {passage.metadata['page'] + 1}"
            )
            st.text(passage.page_content)


st.set_page_config(page_title="CV assistant", page_icon=":material/description:")
st.title("Ask your CVs")
st.write("Upload CVs, explore experience, and compare skills with answers backed by sources.")

for key, default in {"upload_signature": None, "cv_index": None, "messages": [], "page_count": 0}.items():
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("Your CVs")
    uploads = st.file_uploader(
        "Upload one or more CVs", type=["pdf"], accept_multiple_files=True,
        max_upload_size=20, key="cv_uploads",
        help="Text-based PDFs, up to 20 MB each. Scanned PDFs need OCR first.",
    )
    files = [(upload.name, upload.getvalue()) for upload in uploads]
    signature = tuple((name, hashlib.sha256(data).hexdigest()) for name, data in files)
    if signature != st.session_state.upload_signature:
        st.session_state.upload_signature = signature
        st.session_state.cv_index = None
        st.session_state.messages = []
        st.session_state.page_count = 0
        st.session_state.pop("cv_scope", None)
        prepare_cvs.clear()

    process = st.button("Prepare CVs", type="primary", disabled=not files, key="prepare")
    st.caption("Questions and relevant passages are sent to Groq. Uploads and chat stay in this browser session on the server.")
    if st.button("Clear conversation", disabled=not st.session_state.messages, key="clear_chat"):
        st.session_state.messages = []

if process:
    try:
        with st.spinner("Reading your CVs and preparing search..."):
            index, page_count, _ = prepare_cvs(files)
        st.session_state.cv_index = index
        st.session_state.page_count = page_count
        st.session_state.messages = []
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Could not prepare the CVs. Check the PDFs and your connection for the initial model download, then retry.")

ready = st.session_state.cv_index is not None
if ready:
    st.success(f"{len(files)} CV(s) ready · {st.session_state.page_count} pages")
    scope = st.selectbox(
        "Ask about", ["All CVs"] + [f"CV {i}: {name}" for i, (name, _) in enumerate(files, 1)],
        key="cv_scope",
    )
else:
    scope = "All CVs"
    st.info("Upload your PDFs in the sidebar, then select Prepare CVs to start.")

st.caption("Ask a complete question each time, for example: Which CVs mention Python, and what experience do they describe?")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "scope" in message:
            st.caption(f"Asked about: {message['scope']}")
        if "passages" in message:
            show_sources(message["passages"])

question = st.chat_input("Ask a question about your CVs", disabled=not ready)
if question and question.strip():
    st.session_state.messages.append({"role": "user", "content": question, "scope": scope})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        stage = "configuration"
        try:
            with st.spinner("Finding evidence and writing an answer..."):
                llm = create_llm()
                stage = "retrieval"
                index = st.session_state.cv_index
                selected = range(1, len(files) + 1) if scope == "All CVs" else [int(scope.split(":", 1)[0].split()[1])]
                passages = []
                # Retrieve from each selected CV so comparisons include every file.
                for document_id in selected:
                    passages.extend(index.similarity_search(
                        question, k=2,
                        filter=lambda doc, target=str(document_id): doc.metadata["document_id"] == target,
                    ))
                stage = "answer"
                answer = answer_question(llm, question, passages)
            st.markdown(answer)
            show_sources(passages)
            st.session_state.messages.append({"role": "assistant", "content": answer, "passages": passages})
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            if stage == "retrieval":
                st.error(f"CV search failed ({type(exc).__name__}). Prepare the CVs again and retry.")
            else:
                st.error(groq_error_message(exc))
