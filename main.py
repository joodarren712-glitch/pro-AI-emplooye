from __future__ import annotations

import streamlit as st

from services.ai_trainer import format_citations, retrieve_relevant_docs
from services.groq_client import DEFAULT_MODEL, generate_answer, generate_json
from services.storage import execute, fetchall, fetchone, init_db


def _load_docs() -> list[dict]:
    return [dict(row) for row in fetchall("SELECT * FROM knowledge_docs ORDER BY id DESC")]


def _save_doc(title: str, content: str, file_type: str) -> int:
    latest = fetchone(
        "SELECT version FROM knowledge_docs WHERE title = ? ORDER BY version DESC LIMIT 1",
        (title,),
    )
    version = int(latest["version"]) + 1 if latest else 1
    return execute(
        "INSERT INTO knowledge_docs(title, content, file_type, version) VALUES (?, ?, ?, ?)",
        (title, content, file_type, version),
    )


def render_knowledge_library() -> None:
    st.subheader("Knowledge Library")
    with st.form("upload_doc"):
        title = st.text_input("Judul dokumen")
        content = st.text_area("Isi dokumen/SOP", height=180)
        file_type = st.selectbox("Format", ["txt", "pdf", "doc", "ppt"])
        submitted = st.form_submit_button("Simpan Dokumen")
        if submitted:
            if len(content.strip()) < 30:
                st.error("Isi dokumen minimal 30 karakter.")
            else:
                doc_id = _save_doc(title=title.strip(), content=content.strip(), file_type=file_type)
                st.success(f"Dokumen tersimpan (id={doc_id}).")

    query = st.text_input("Cari dokumen")
    if query:
        rows = fetchall(
            "SELECT * FROM knowledge_docs WHERE title LIKE ? OR content LIKE ? ORDER BY id DESC",
            (f"%{query}%", f"%{query}%"),
        )
    else:
        rows = fetchall("SELECT * FROM knowledge_docs ORDER BY id DESC")

    for row in rows:
        doc = dict(row)
        with st.expander(f"{doc['title']} (v{doc['version']})"):
            st.caption(f"Format: {doc['file_type']} | ID: {doc['id']}")
            st.write(doc["content"])


def render_chat() -> None:
    st.subheader("AI Trainer Chat (Groq)")
    docs = _load_docs()
    model = st.text_input("Model Groq", value=DEFAULT_MODEL)
    question = st.text_area("Pertanyaan karyawan", placeholder="Bagaimana prosedur refund produk?")

    if st.button("Tanya AI", type="primary"):
        if not question.strip():
            st.error("Pertanyaan wajib diisi.")
            return

        relevant = retrieve_relevant_docs(question, docs)
        if not relevant:
            st.warning("AI tidak menjawab karena tidak ada dokumen sumber yang relevan.")
            return

        context = "\n\n".join([f"[{d['id']}] {d['title']}\n{d['content'][:1200]}" for d in relevant])
        system_prompt = (
            "Anda adalah AI trainer perusahaan. Jawab HANYA berdasarkan konteks dokumen yang diberikan. "
            "Jika informasi kurang, katakan tidak ada sumber cukup. Jawab dalam Bahasa Indonesia."
        )
        user_prompt = f"Konteks dokumen:\n{context}\n\nPertanyaan: {question}\n\nBerikan jawaban ringkas dan praktis."

        try:
            answer = generate_answer(system_prompt=system_prompt, user_prompt=user_prompt, model=model)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal memanggil Groq: {exc}")
            return

        st.markdown("### Jawaban AI")
        st.write(answer)
        st.info(format_citations(relevant))


def render_module_generator() -> None:
    st.subheader("Training Module Generator (Groq)")
    docs = _load_docs()
    role = st.selectbox("Role target", ["employee", "trainer", "hr", "admin", "supervisor", "cashier"])
    selected_ids = st.multiselect("Pilih dokumen sumber", [d["id"] for d in docs], format_func=lambda i: next(d["title"] for d in docs if d["id"] == i))

    if st.button("Generate Module"):
        selected_docs = [d for d in docs if d["id"] in selected_ids] if selected_ids else docs[:3]
        if not selected_docs:
            st.error("Belum ada dokumen knowledge.")
            return

        context = "\n\n".join([f"{d['title']}\n{d['content'][:1400]}" for d in selected_docs])
        system_prompt = "Anda adalah instruktur corporate learning. Buat modul training yang praktis, jelas, dan sesuai SOP."
        user_prompt = (
            f"Role: {role}\nKonteks:\n{context}\n\n"
            "Buat output berformat:\n"
            "Judul: ...\n"
            "Tujuan Belajar:\n- ...\n"
            "Materi:\n1. ...\n2. ...\n3. ...\n"
            "Checklist praktik:\n- ..."
        )

        try:
            generated = generate_answer(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal generate module: {exc}")
            return

        title = f"Auto Module - {role.title()}"
        module_id = execute(
            "INSERT INTO training_modules(title, role, content) VALUES (?, ?, ?)",
            (title, role, generated),
        )
        st.success(f"Module tersimpan (id={module_id}).")
        st.write(generated)

    st.markdown("### Daftar Module")
    rows = fetchall("SELECT * FROM training_modules ORDER BY id DESC")
    for row in rows:
        m = dict(row)
        with st.expander(f"{m['title']} ({m['role']})"):
            st.write(m["content"])


def render_quiz_generator() -> None:
    st.subheader("Quiz Generator (Groq)")
    rows = [dict(r) for r in fetchall("SELECT * FROM training_modules ORDER BY id DESC")]
    if not rows:
        st.info("Belum ada module. Buat module dulu.")
        return

    module_id = st.selectbox("Pilih module", [m["id"] for m in rows], format_func=lambda i: next(m["title"] for m in rows if m["id"] == i))
    n_questions = st.slider("Jumlah soal", 3, 10, 5)

    if st.button("Generate Quiz"):
        module = next(m for m in rows if m["id"] == module_id)
        system_prompt = "Anda membuat kuis training karyawan. Output HARUS JSON valid."
        user_prompt = (
            "Buat quiz multiple choice dalam JSON valid dengan skema:\n"
            "{\"questions\": [{\"question\": str, \"options\": [str,str,str,str], \"correct_index\": int, \"explanation\": str}]}\n"
            f"Jumlah soal: {n_questions}\n"
            f"Konten module:\n{module['content'][:2500]}"
        )
        try:
            quiz = generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal generate quiz: {exc}")
            return

        st.json(quiz)


def main() -> None:
    st.set_page_config(page_title="TrainAI - Streamlit + Groq", layout="wide")
    st.title("TrainAI - AI Employee Training")
    st.caption("Versi Streamlit dengan AI asli dari Groq")

    init_db()

    with st.sidebar:
        st.header("Pengaturan")
        st.markdown("Set environment variable `GROQ_API_KEY` sebelum menjalankan app.")
        section = st.radio(
            "Pilih fitur",
            ["Knowledge Library", "AI Trainer Chat", "Training Module", "Quiz Generator"],
        )

    if section == "Knowledge Library":
        render_knowledge_library()
    elif section == "AI Trainer Chat":
        render_chat()
    elif section == "Training Module":
        render_module_generator()
    else:
        render_quiz_generator()


if __name__ == "__main__":
    main()