import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="CSV Similarity Matcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📑 CSV Similarity Matcher")
st.markdown(
    "Upload two CSV files, select columns to compare, and find the most similar rows using embeddings & FAISS."
)

# Sidebar for settings
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Sentence Transformer Model", 
    options=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    index=0
)
top_k = st.sidebar.slider("Number of nearest matches (k)", min_value=1, max_value=5, value=1)
show_hist = st.sidebar.checkbox("Show similarity histogram", value=True)

# File upload
col1, col2 = st.columns(2)
with col1:
    origin_file = st.file_uploader("Upload origin.csv", type=["csv"] )
with col2:
    dest_file = st.file_uploader("Upload destination.csv", type=["csv"] )

# Main processing
if origin_file and dest_file:
    # Read CSVs
    origin_df = pd.read_csv(origin_file)
    dest_df = pd.read_csv(dest_file)

    st.success(f"Loaded origin ({len(origin_df)} rows) and destination ({len(dest_df)} rows)")

    # Find common columns
    common_cols = list(set(origin_df.columns) & set(dest_df.columns))
    if not common_cols:
        st.error("No common columns found between the two CSVs.")
    else:
        cols = st.multiselect(
            "Select columns to include for similarity matching:",
            options=common_cols,
            default=common_cols[:2]
        )

        if cols:
            if st.button("Run matching"):
                with st.spinner("Generating embeddings and searching..."):
                    # Combine text
                    origin_texts = origin_df[cols].fillna('').astype(str).agg(' '.join, axis=1).tolist()
                    dest_texts = dest_df[cols].fillna('').astype(str).agg(' '.join, axis=1).tolist()

                    # Load model (cached)
                    @st.cache_resource
                    def load_model(name):
                        return SentenceTransformer(name)
                    model = load_model(model_name)

                    # Encode
                    @st.cache_data
                    def encode_texts(texts):
                        return model.encode(texts, show_progress_bar=True)
                    origin_embeds = encode_texts(origin_texts)
                    dest_embeds = encode_texts(dest_texts)

                    # FAISS index
                    dim = origin_embeds.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(dest_embeds.astype('float32'))

                    D, I = index.search(origin_embeds.astype('float32'), k=top_k)
                    similarity = 1 - (D / np.max(D))

                    # Build results
                    for match_i in range(top_k):
                        origin_url = origin_df['Address'] if 'Address' in origin_df.columns else None
                        matched = dest_df.iloc[I[:, match_i]]
                        scores = np.round(similarity[:, match_i], 4)

                        result_df = pd.DataFrame({
                            'origin_index': origin_df.index,
                            'matched_index': I[:, match_i],
                            'similarity_score': scores
                        })
                        if 'Address' in origin_df.columns and 'Address' in dest_df.columns:
                            result_df['origin_address'] = origin_df['Address'].values
                            result_df['matched_address'] = matched['Address'].values

                        st.subheader(f"Top {match_i+1} Matches")
                        st.dataframe(result_df)
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download top {match_i+1} matches as CSV",
                            data=csv,
                            file_name=f"matches_top{match_i+1}.csv",
                            mime="text/csv"
                        )

                    if show_hist:
                        # Plot histogram
                        fig, ax = plt.subplots()
                        ax.hist(similarity.flatten(), bins=20)
                        ax.set_xlabel('Similarity Score')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Distribution of Similarity Scores')
                        st.pyplot(fig)
        else:
            st.info("Select at least one column to proceed.")
