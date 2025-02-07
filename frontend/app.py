import requests
import streamlit as st

# Backend API URL
API_URL = "http://127.0.0.1:8000/search"  # Adjust if hosted elsewhere

st.set_page_config(page_title="Semantic Search", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    .search-box {
        padding: 10px;
        font-size: 20px;
        border-radius: 10px;
    }
    .result-card {
        background-color: #f9f9f9;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Semantic Search UI")
st.write("Enter a query below and get relevant documents from the database.")

# User Input
query = st.text_input("Enter your search query:", "", placeholder="Type something...")
top_k = st.slider("Number of results:", 1, 10, 5)

# Search Button
if st.button("Search", type="primary"):
    if query.strip():
        with st.spinner("Searching..."):
            try:
                response = requests.post(API_URL, json={"text": query, "top_k": top_k})
                response.raise_for_status()
                results = response.json().get("documents", [])

                if results:
                    st.success(f"🔹 Found {len(results)} relevant documents.")
                    for idx, doc in enumerate(results):
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <b>🔹 Result {idx + 1}:</b> {doc}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("⚠ No relevant results found. Try another query.")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠ Please enter a valid query.")
