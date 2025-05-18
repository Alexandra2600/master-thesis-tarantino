import streamlit as st
import asyncio
from datetime import datetime

import KG_construction
import GraphRAG
import utils

# Get current date in yyyy/mm/dd format
current_date = datetime.today().strftime("%Y/%m/%d")

# =========================
# STYLING
# =========================
st.markdown("""
    <style>
    .stApp {
        background-color: #fdf8f3;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #309799;
        font-size: 42px;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .stMarkdown p {
        color: #333;
        font-size: 18px;
        text-align: center;
    }
    textarea {
        background-color: #ffffff !important;
        border: 1px solid #ccc !important;
        border-radius: 10px !important;
        padding: 10px !important;
        color: #000 !important;
        font-size: 16px !important;
    }
    button[kind="primary"] {
        background-color: #309799;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        margin-top: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        color: #000000;
        font-weight: bold;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #fdf8f3;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown("""
    <h1 style='color: #309799; font-size: 50px; text-align: center;'>
        Digital Diary for MCI
    </h1>
""", unsafe_allow_html=True)

st.write(f"📅 <b>Today is:</b> {current_date}", unsafe_allow_html=True)


# =========================
# TABS
# =========================
tab_insert, tab_query = st.tabs(["📝 Add Information", "🔍 Ask Information"])


# =========================
# TAB 1: Data Insertion
# =========================
with tab_insert:
    user_input = st.text_area("Input", label_visibility="collapsed", placeholder="Write your task here...", height=100)
    
    if st.button("📌 Save this"):
        # Normalize temporal references and personalize pronouns
        user_input = utils.process_text(text=user_input, current_date=current_date, user_name="Samuel")
        
        # Add to KG 
        response = asyncio.run(KG_construction.add_user_input_to_kg(user_input))
        st.success(response)

        # Entity resolution to avoid duplicates
        resolved = asyncio.run(KG_construction.resolve_kg_entities())


# =========================
# TAB 2: Question Answering
# =========================
with tab_query:
    question = st.text_area("Query", label_visibility="collapsed", placeholder="Write your question here...", height=100)
    
    if st.button("🔎 Get an Answer"):
        # Normalize temporal references in the question
        question = utils.process_date(text=question, current_date=current_date)

        # Use GraphRAG to generate answer
        answer = GraphRAG.query_graph_with_context(question)

        st.markdown(f"""
            <div style='background-color: #ffffff;
                        padding: 15px;
                        border-radius: 10px;
                        color: #000000;
                        font-size: 16px;
                        text-align: left;
                        margin-left: 0;
                        margin-right: auto;
                        width: fit-content;'>
                💬 <strong>Answer:</strong> {answer}
            </div>
        """, unsafe_allow_html=True)