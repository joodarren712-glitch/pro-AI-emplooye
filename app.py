"""
TrainAI - Streamlit App with Groq AI
Training Platform dengan AI-powered features
"""

import sqlite3
import json
import re
import math
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Optional

import streamlit as st
from groq import Groq

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = Path(__file__).parent / "trainai.db"
GROQ_MODEL = "llama-3.1-8b-instant"

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS knowledge_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            file_type TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1
        );
        
        CREATE TABLE IF NOT EXISTS training_modules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_id INTEGER NOT NULL,
            questions_json TEXT NOT NULL,
            FOREIGN KEY(module_id) REFERENCES training_modules(id)
        );
        
        CREATE TABLE IF NOT EXISTS progress (
            user_id INTEGER NOT NULL,
            module_id INTEGER NOT NULL,
            completion_percent REAL NOT NULL,
            latest_score REAL,
            PRIMARY KEY (user_id, module_id),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(module_id) REFERENCES training_modules(id)
        );
    """)
    conn.commit()
    conn.close()


def execute(query: str, values: tuple = ()) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id


def fetchall(query: str, values: tuple = ()) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, values)
    rows = cursor.fetchall()
    conn.close()
    return rows


def fetchone(query: str, values: tuple = ()) -> Optional[sqlite3.Row]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, values)
    row = cursor.fetchone()
    conn.close()
    return row


# ============================================================================
# AI FUNCTIONS (Groq)
# ============================================================================

def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def tokenize(text: str) -> list[str]:
    """Tokenize text for similarity matching"""
    return [token.lower() for token in re.findall(r"\w+", text)]


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    """Calculate cosine similarity between two token counters"""
    common = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a + norm_b)


def retrieve_relevant_docs(question: str, docs: list[dict], top_k: int = 3) -> list[dict]:
    """Retrieve relevant documents using cosine similarity"""
    question_vec = Counter(tokenize(question))
    scored = []
    
    for doc in docs:
        doc_vec = Counter(tokenize(doc["content"]))
        similarity = cosine_similarity(question_vec, doc_vec)
        if similarity > 0:
            scored.append((similarity, doc))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def groq_chat_response(client: Groq, question: str, context: str) -> str:
    """Get AI response from Groq"""
    system_prompt = """Anda adalah asisten AI yang membantu karyawan memahami SOP dan prosedur perusahaan.
Jawab dengan jelas, profesional, dan merujuk pada dokumen SOP yang tersedia."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Konteks SOP:\n{context}\n\nPertanyaan: {question}"}
    ]
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content


def groq_generate_module(client: Groq, role: str, content: str) -> dict:
    """Generate training module using Groq AI"""
    prompt = f"""Buat modul training untuk role '{role}' berdasarkan konten berikut:

{content}

Format output JSON dengan struktur:
{{
    "title": "Judul Modul",
    "learning_objectives": ["obj1", "obj2", "obj3"],
    "content": "Konten modul lengkap",
    "duration_hours": 2
}}"""
    
    messages = [
        {"role": "system", "content": "Anda adalah instructional designer yang ahli membuat modul training."},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=800
    )
    
    # Extract JSON from response
    content_text = response.choices[0].message.content
    try:
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Fallback
    return {
        "title": f"Training Module - {role.title()}",
        "learning_objectives": ["Pahami SOP", "Praktikkan prosedur", "Kuasai kompetensi"],
        "content": content_text,
        "duration_hours": 2
    }


def groq_generate_quiz(client: Groq, module_content: str, num_questions: int = 5) -> list[dict]:
    """Generate quiz questions using Groq AI"""
    prompt = f"""Buat {num_questions} pertanyaan quiz berdasarkan konten modul berikut:

{module_content}

Format output JSON array:
[
    {{
        "question": "Pertanyaan...",
        "options": ["A", "B", "C", "D"],
        "correct_index": 0
    }}
]"""
    
    messages = [
        {"role": "system", "content": "Anda adalah evaluator yang membuat quiz training."},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    
    content_text = response.choices[0].message.content
    try:
        json_match = re.search(r'\[.*\]', content_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Fallback - simple quiz
    return [
        {
            "question": f"Pertanyaan {i+1}: Apa poin penting dari modul ini?",
            "options": ["Ikuti SOP", "Lewati prosedur", "Serahkan ke supervisor", "Abaikan"],
            "correct_index": 0
        }
        for i in range(num_questions)
    ]


def groq_evaluate_scenario(client: Groq, scenario: str, response: str) -> dict:
    """Evaluate scenario response using Groq AI"""
    prompt = f"""Evaluasi respons karyawan terhadap scenario berikut:

SCENARIO: {scenario}
RESPONS KARYAWAN: {response}

Berikan evaluasi dengan format JSON:
{{
    "score": 75,
    "feedback": "Feedback konstruktif",
    "recommendation": "Rekomendasi perbaikan"
}}

Score 0-100 berdasarkan: empati, kepatuhan SOP, efektivitas solusi."""
    
    messages = [
        {"role": "system", "content": "Anda adalah trainer yang mengevaluasi respons karyawan."},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=400
    )
    
    content_text = response.choices[0].message.content
    try:
        json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    return {
        "score": 70,
        "feedback": "Respons sudah baik, tingkatkan empati.",
        "recommendation": "Gunakan format: empati -> verifikasi -> solusi -> konfirmasi."
    }


# ============================================================================
# STREAMLIT UI
# ============================================================================

def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        init_db()
    
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""


def sidebar():
    with st.sidebar:
        st.title("🎓 TrainAI")
        st.markdown("---")
        
        # Groq API Key
        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value=st.session_state.groq_api_key,
            help="Dapatkan dari https://console.groq.com/keys"
        )
        if api_key:
            st.session_state.groq_api_key = api_key
        
        st.markdown("---")
        
        # User Selection
        users = fetchall("SELECT * FROM users ORDER BY id DESC")
        
        if users:
            user_options = {f"{u['name']} ({u['role']})": u for u in users}
            selected = st.selectbox(
                "👤 Pilih User",
                options=list(user_options.keys())
            )
            st.session_state.current_user = user_options[selected]
        else:
            st.info("Belum ada user. Buat user baru di bawah.")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "📋 Menu",
            ["Dashboard", "Knowledge Base", "AI Chat", "Training Modules", "Quiz", "Scenario Training"],
            index=0
        )
        
        return page


def create_user_form():
    with st.expander("➕ Buat User Baru"):
        with st.form("create_user"):
            name = st.text_input("Nama")
            email = st.text_input("Email")
            role = st.selectbox(
                "Role",
                ["employee", "trainer", "hr", "admin", "supervisor", "cashier"]
            )
            
            submitted = st.form_submit_button("Create User", use_container_width=True)
            if submitted and name and email:
                try:
                    execute("INSERT INTO users(name, email, role) VALUES (?, ?, ?)",
                           (name, email, role))
                    st.success(f"User {name} berhasil dibuat!")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Email sudah terdaftar!")


def page_dashboard():
    st.title("📊 Dashboard")
    
    # Stats
    users = fetchall("SELECT COUNT(*) as total FROM users")
    docs = fetchall("SELECT COUNT(*) as total FROM knowledge_docs")
    modules = fetchall("SELECT COUNT(*) as total FROM training_modules")
    progress = fetchall("SELECT * FROM progress")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", users[0]["total"] if users else 0)
    with col2:
        st.metric("Knowledge Docs", docs[0]["total"] if docs else 0)
    with col3:
        st.metric("Training Modules", modules[0]["total"] if modules else 0)
    
    if progress:
        avg_completion = sum(p["completion_percent"] for p in progress) / len(progress)
        avg_score = sum(p["latest_score"] or 0 for p in progress) / len(progress)
        
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Avg Completion", f"{avg_completion:.1f}%")
        with col5:
            st.metric("Avg Quiz Score", f"{avg_score:.1f}")
    
    st.markdown("---")
    
    # Recent Activity
    st.subheader("📚 Recent Knowledge Docs")
    recent_docs = fetchall("SELECT * FROM knowledge_docs ORDER BY id DESC LIMIT 5")
    if recent_docs:
        for doc in recent_docs:
            st.markdown(f"📄 **{doc['title']}** (v{doc['version']}) - {doc['file_type']}")
    else:
        st.info("Belum ada dokumen.")


def page_knowledge_base():
    st.title("📚 Knowledge Base")
    
    # Upload new document
    with st.expander("➕ Upload Dokumen Baru"):
        with st.form("upload_doc"):
            title = st.text_input("Judul Dokumen")
            content = st.text_area("Konten", height=200)
            file_type = st.selectbox("Tipe File", ["txt", "pdf", "doc", "ppt"])
            
            submitted = st.form_submit_button("Upload", use_container_width=True)
            if submitted and title and len(content) >= 30:
                # Check version
                latest = fetchone(
                    "SELECT version FROM knowledge_docs WHERE title = ? ORDER BY version DESC LIMIT 1",
                    (title,)
                )
                version = int(latest["version"]) + 1 if latest else 1
                
                execute("INSERT INTO knowledge_docs(title, content, file_type, version) VALUES (?, ?, ?, ?)",
                       (title, content, file_type, version))
                st.success(f"Dokumen '{title}' berhasil diupload!")
                st.rerun()
            elif submitted:
                st.error("Konten minimal 30 karakter!")
    
    st.markdown("---")
    
    # List documents
    search = st.text_input("🔍 Cari dokumen...")
    
    if search:
        docs = fetchall(
            "SELECT * FROM knowledge_docs WHERE title LIKE ? OR content LIKE ? ORDER BY id DESC",
            (f"%{search}%", f"%{search}%")
        )
    else:
        docs = fetchall("SELECT * FROM knowledge_docs ORDER BY id DESC")
    
    if docs:
        for doc in docs:
            with st.container():
                st.markdown(f"### 📄 {doc['title']}")
                st.caption(f"Version {doc['version']} | Type: {doc['file_type']}")
                with st.expander("Lihat konten"):
                    st.write(doc["content"])
                st.divider()
    else:
        st.info("Belum ada dokumen.")


def page_ai_chat():
    st.title("💬 AI Chat Assistant")
    
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Masukkan Groq API Key di sidebar untuk menggunakan AI Chat")
        return
    
    if not st.session_state.current_user:
        st.warning("⚠️ Pilih user di sidebar terlebih dahulu")
        return
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Tanya tentang SOP..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get relevant docs
        docs = fetchall("SELECT * FROM knowledge_docs")
        docs_list = [dict(row) for row in docs]
        
        if docs_list:
            relevant = retrieve_relevant_docs(prompt, docs_list)
            
            if relevant:
                context = "\n\n".join([d["content"][:300] for d in relevant])
                
                with st.chat_message("assistant"):
                    with st.spinner("AI sedang menjawab..."):
                        try:
                            client = get_groq_client(st.session_state.groq_api_key)
                            answer = groq_chat_response(client, prompt, context)
                            st.write(answer)
                            
                            # Show citations
                            if relevant:
                                with st.expander("📎 Sumber"):
                                    for doc in relevant:
                                        st.markdown(f"**{doc['title']}**")
                                        st.caption(doc["content"][:150] + "...")
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.info("Tidak menemukan dokumen relevan.")
                st.session_state.chat_history.append({"role": "assistant", "content": "Maaf, tidak menemukan informasi relevan di knowledge base."})
        else:
            st.info("Upload dokumen knowledge base terlebih dahulu.")


def page_training_modules():
    st.title("📖 Training Modules")
    
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Masukkan Groq API Key di sidebar untuk menggunakan AI features")
    
    # Create module form
    with st.expander("➕ Buat Module Manual"):
        with st.form("create_module"):
            title = st.text_input("Judul Module")
            role = st.selectbox("Role", ["employee", "trainer", "hr", "admin", "supervisor", "cashier"])
            content = st.text_area("Konten", height=150)
            
            submitted = st.form_submit_button("Create Module", use_container_width=True)
            if submitted and title and content:
                execute("INSERT INTO training_modules(title, role, content) VALUES (?, ?, ?)",
                       (title, role, content))
                st.success("Module berhasil dibuat!")
                st.rerun()
    
    # AI Generate Module
    if st.session_state.groq_api_key:
        with st.expander("✨ Generate Module dengan AI"):
            docs = fetchall("SELECT * FROM knowledge_docs ORDER BY id DESC")
            
            if docs:
                with st.form("generate_module"):
                    role = st.selectbox("Role Target", ["employee", "trainer", "hr", "admin", "supervisor", "cashier"])
                    selected_docs = st.multiselect(
                        "Pilih Dokumen Sumber",
                        options=[f"{d['id']}: {d['title']}" for d in docs]
                    )
                    
                    submitted = st.form_submit_button("Generate dengan AI", use_container_width=True)
                    if submitted:
                        # Extract doc IDs
                        doc_ids = [int(s.split(":")[0]) for s in selected_docs]
                        source_docs = [dict(d) for d in docs if d["id"] in doc_ids] if doc_ids else [dict(d) for d in docs[:3]]
                        
                        if source_docs:
                            content = "\n\n".join([d["content"] for d in source_docs])
                            
                            with st.spinner("AI sedang generate module..."):
                                try:
                                    client = get_groq_client(st.session_state.groq_api_key)
                                    module_data = groq_generate_module(client, role, content)
                                    
                                    # Save to DB
                                    module_content = f"""Learning Objectives:
{chr(10).join('- ' + obj for obj in module_data.get('learning_objectives', []))}

{module_data.get('content', '')}"""
                                    
                                    execute("INSERT INTO training_modules(title, role, content) VALUES (?, ?, ?)",
                                           (module_data["title"], role, module_content))
                                    st.success(f"Module '{module_data['title']}' berhasil dibuat!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
            else:
                st.info("Upload dokumen knowledge base terlebih dahulu.")
    
    st.markdown("---")
    
    # List modules
    role_filter = st.selectbox("Filter by Role", ["all", "employee", "trainer", "hr", "admin", "supervisor", "cashier"])
    
    if role_filter == "all":
        modules = fetchall("SELECT * FROM training_modules ORDER BY id DESC")
    else:
        modules = fetchall("SELECT * FROM training_modules WHERE role = ? ORDER BY id DESC", (role_filter,))
    
    if modules:
        for mod in modules:
            with st.container():
                st.markdown(f"### 📖 {mod['title']}")
                st.caption(f"Role: {mod['role']}")
                with st.expander("Lihat konten"):
                    st.write(mod["content"])
                st.divider()
    else:
        st.info("Belum ada module.")


def page_quiz():
    st.title("📝 Quiz & Assessment")
    
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Masukkan Groq API Key di sidebar untuk menggunakan AI features")
    
    if not st.session_state.current_user:
        st.warning("⚠️ Pilih user di sidebar terlebih dahulu")
        return
    
    user = st.session_state.current_user
    
    # Generate Quiz
    with st.expander("✨ Generate Quiz dengan AI"):
        modules = fetchall("SELECT * FROM training_modules ORDER BY id DESC")
        
        if modules:
            with st.form("generate_quiz"):
                selected_module = st.selectbox(
                    "Pilih Module",
                    options=[f"{m['id']}: {m['title']}" for m in modules]
                )
                num_questions = st.slider("Jumlah Pertanyaan", 3, 10, 5)
                
                submitted = st.form_submit_button("Generate Quiz", use_container_width=True)
                if submitted:
                    module_id = int(selected_module.split(":")[0])
                    module = fetchone("SELECT * FROM training_modules WHERE id = ?", (module_id,))
                    
                    if module:
                        with st.spinner("AI sedang generate quiz..."):
                            try:
                                client = get_groq_client(st.session_state.groq_api_key)
                                questions = groq_generate_quiz(client, dict(module)["content"], num_questions)
                                
                                # Save to session for submission
                                st.session_state.current_quiz = {
                                    "module_id": module_id,
                                    "questions": questions
                                }
                                st.success("Quiz berhasil di-generate!")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
        else:
            st.info("Belum ada module training.")
    
    st.markdown("---")
    
    # Take Quiz
    if "current_quiz" in st.session_state:
        quiz = st.session_state.current_quiz
        
        st.subheader("📝 Kerjakan Quiz")
        
        with st.form("take_quiz"):
            answers = []
            for i, q in enumerate(quiz["questions"]):
                st.markdown(f"**{i+1}. {q['question']}**")
                answer = st.radio(
                    "Pilih jawaban:",
                    q["options"],
                    key=f"q_{i}",
                    index=None
                )
                # Find index of selected option
                if answer:
                    answers.append(q["options"].index(answer))
                else:
                    answers.append(-1)
            
            submitted = st.form_submit_button("Submit Quiz", use_container_width=True)
            if submitted:
                # Calculate score
                correct = sum(1 for i, q in enumerate(quiz["questions"]) if answers[i] == q["correct_index"])
                total = len(quiz["questions"])
                score = round((correct / total) * 100, 2)
                
                # Save progress
                execute(
                    "INSERT OR REPLACE INTO progress(user_id, module_id, completion_percent, latest_score) VALUES (?, ?, ?, ?)",
                    (user["id"], quiz["module_id"], 100.0, score)
                )
                
                st.success(f"Quiz selesai! Score: {score}% ({correct}/{total} benar)")
                
                # Clear quiz
                del st.session_state.current_quiz
    
    # Progress
    st.markdown("---")
    st.subheader("📊 Progress Saya")
    
    progress = fetchall("SELECT * FROM progress WHERE user_id = ?", (user["id"],))
    
    if progress:
        for p in progress:
            st.progress(p["completion_percent"] / 100)
            st.caption(f"Module {p['module_id']}: {p['completion_percent']}% - Score: {p['latest_score']}")
    else:
        st.info("Belum ada progress.")


def page_scenario_training():
    st.title("🎭 Scenario Training")
    
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Masukkan Groq API Key di sidebar untuk menggunakan AI features")
    
    if not st.session_state.current_user:
        st.warning("⚠️ Pilih user di sidebar terlebih dahulu")
        return
    
    user = st.session_state.current_user
    
    # Scenario input
    st.subheader("📖 Scenario")
    
    scenarios = {
        "Complaint Handling": "Pelanggan marah karena order terlambat 3 hari dan meminta refund.",
        "Product Inquiry": "Pelanggan bertanya tentang spesifikasi produk yang kompleks.",
        "Technical Issue": "Sistem error saat pelanggan ingin checkout.",
        "Escalation": "Pelanggan meminta bicara dengan supervisor."
    }
    
    selected_scenario = st.selectbox("Pilih Scenario", list(scenarios.keys()))
    scenario_text = scenarios[selected_scenario]
    
    st.info(scenario_text)
    
    # Response input
    st.subheader("💬 Respons Anda")
    
    response = st.text_area(
        "Tulis respons Anda terhadap scenario di atas:",
        height=150,
        placeholder="Contoh: Maaf atas ketidaknyamanan ini, saya akan bantu cek..."
    )
    
    if st.button("Evaluate Respons", use_container_width=True):
        if response:
            if st.session_state.groq_api_key:
                with st.spinner("AI sedang evaluasi..."):
                    try:
                        client = get_groq_client(st.session_state.groq_api_key)
                        result = groq_evaluate_scenario(client, scenario_text, response)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Score", result.get("score", 0))
                        
                        with col2:
                            if result.get("score", 0) >= 70:
                                st.success("✅ Good!")
                            else:
                                st.warning("⚠️ Perlu perbaikan")
                        
                        st.markdown("### Feedback")
                        st.write(result.get("feedback", ""))
                        
                        st.markdown("### Recommendation")
                        st.write(result.get("recommendation", ""))
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Masukkan Groq API Key untuk evaluasi dengan AI")
        else:
            st.warning("Tulis respons terlebih dahulu")


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="TrainAI",
        page_icon="🎓",
        layout="wide"
    )
    
    init_session_state()
    
    page = sidebar()
    
    create_user_form()
    
    if page == "Dashboard":
        page_dashboard()
    elif page == "Knowledge Base":
        page_knowledge_base()
    elif page == "AI Chat":
        page_ai_chat()
    elif page == "Training Modules":
        page_training_modules()
    elif page == "Quiz":
        page_quiz()
    elif page == "Scenario Training":
        page_scenario_training()


if __name__ == "__main__":
    main()
