# app.py
import ast  # for parsing the image list string
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/flipkart-products/flipkart_com-ecommerce_sample.csv"
EMB_PATH = "data/flipkart_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data_and_embeddings():
    df = pd.read_csv(DATA_PATH)
    embeddings = np.load(EMB_PATH)
    # normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    emb_norm = embeddings / norms
    return df, emb_norm

def search_products(query: str, df: pd.DataFrame, emb_norm: np.ndarray, model, top_k: int = 10):
    # embed the query
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    # cosine similarity
    scores = emb_norm @ q_emb
    # top k
    top_k = min(top_k, len(df))
    idx = np.argsort(-scores)[:top_k]
    return df.iloc[idx].reset_index(drop=True), scores[idx]

def get_first_image_url(row):
    """
    Parse the `image` column which looks like:
    ["http://url1", "http://url2", ...]
    and return the first URL, or None.
    """
    images = row.get("image", "")
    if not isinstance(images, str) or not images.strip():
        return None
    try:
        urls = ast.literal_eval(images)
        if isinstance(urls, list) and len(urls) > 0:
            return urls[0]
    except Exception:
        return None
    return None

def main():
    st.set_page_config(page_title="Flipkart Product Q&A", page_icon="🛒")
    st.title("🛒 Local semantic search over Flipkart products")

    st.write(
        "Ask anything about the products and I will show the most relevant rows\n"
        "Examples:\n"
        "- 'solid women cycling shorts'\n"
        "- 'cheap cotton lycra women's shorts under 500'\n"
        "- 'Alisha brand navy shorts'\n"
    )

    df, emb_norm = load_data_and_embeddings()
    model = load_model()

    with st.expander("Preview data"):
        st.dataframe(df.head())

    query = st.text_input("Your query", value="solid women cycling shorts")
    top_k = st.slider("Number of results", 3, 30, 10)

    if st.button("Search") and query.strip():
        with st.spinner("Searching..."):
            results, scores = search_products(query, df, emb_norm, model, top_k=top_k)

        if results.empty:
            st.error("No results found.")
            return

        st.subheader("Results")
        for i, (idx, row) in enumerate(results.iterrows()):
            s = scores[i]
            st.markdown(f"### {row.get('product_name', '')}")

            # Show image (first URL from `image` column)
            image_url = get_first_image_url(row)
            if image_url:
                st.image(image_url, width=150)

            st.write(f"Brand: {row.get('brand', '')}")
            st.write(f"Category: {row.get('product_category_tree', '')}")
            st.write(f"Retail price: {row.get('retail_price', '')}")
            st.write(f"Discounted price: {row.get('discounted_price', '')}")
            st.write(f"Rating: {row.get('overall_rating', '')}")
            st.write(f"Description: {row.get('description', '')}")
            st.write(f"Similarity score: `{s:.3f}`")
            url = row.get("product_url", "")
            if isinstance(url, str) and url:
                st.markdown(f"[View product]({url})")
            st.markdown("---")

if __name__ == "__main__":
    main()
