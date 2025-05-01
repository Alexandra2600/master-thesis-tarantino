import streamlit as st
import asyncio
import os
import add_data 
import ask_data
import utility_function 
from datetime import datetime

# Save current date
current_date = datetime.today().strftime("%Y/%m/%d")

def update_user_info():
    """Update user information in session state."""
    st.session_state.user_info["name"] = st.session_state.new_user_name

# Streamlit UI
st.title("Knowledge Graph for TAM in CI - Prototype")

# Interaction Tabs
tab1, tab2, tab3 = st.tabs(["Add", "Ask", "Utility"])

# Tab 1 - Add to Knowledge Graph
with tab1:
    st.subheader("Add data")
    user_input = st.text_area("Input data about user activity")
    if st.button("Add"):
        user_input = utility_function.process_text(text= user_input, current_date=current_date, user_name="Priyanka")
        st.write(f"Processed input: {user_input}")
        # response = asyncio.run(add_data.add_data_to_graph(user_input))
        # st.success(response)
        # res = asyncio.run(add_data.resolve_entities())
        # st.success(res)
        

# Tab 2 - Query the Knowledge Graph
with tab2:
    st.subheader("Query data")
    question = st.text_area("Enter your question")
    if st.button("Ask"):
        #question = utility_function.process_date(text = question, current_date = current_date)
        st.write(f"Processed question: {question}")
        answer = ask_data.ask_data(question)
        st.write(f"Answer: {answer}")

# Tab 3 - Utility
with tab3:
    st.write(f"Current Date: {current_date}")
    
    st.subheader("User Information")
    userInfo = st.text_area("Enter user info")
    if st.button("Add User Info"):
        response = asyncio.run(add_data.add_data_to_graph(userInfo))
        st.success(response)
        
    st.subheader("Resolver Entities")
    if st.button("Resolve Entities"):
        response = asyncio.run(add_data.resolve_entities())
        st.success(response)