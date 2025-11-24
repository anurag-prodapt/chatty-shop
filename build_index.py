# build_index.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/flipkart-products/flipkart_com-ecommerce_sample.csv"
EMB_PATH = "data/flipkart_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_text_column(df: pd.DataFrame) -> list[str]:
    """
    Combine the useful columns into one text string per product.
    Adjust this to taste.
    """
    name = df.get("product_name", "").fillna("")
    category = df.get("product_category_tree", "").fillna("")
    brand = df.get("brand", "").fillna("")
    desc = df.get("description", "").fillna("")

    text = (
        "Name: " + name.astype(str) + ". "
        "Category: " + category.astype(str) + ". "
        "Brand: " + brand.astype(str) + ". "
        "Description: " + desc.astype(str)
    )
    return text.tolist()

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    print("Loading CSV...")
    df = pd.read_csv(DATA_PATH)
    print("Rows:", len(df))

    print("Building text column...")
    texts = build_text_column(df)

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding rows to embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True  # so dot product is cosine similarity
    )
    embeddings = np.array(embeddings, dtype="float32")
    print("Embeddings shape:", embeddings.shape)

    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.save(EMB_PATH, embeddings)
    print("Saved embeddings to", EMB_PATH)

if __name__ == "__main__":
    main()
