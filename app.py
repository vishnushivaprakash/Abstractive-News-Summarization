import streamlit as st
from summarizer import NewsSummarizer

# 1. Page Configuration
st.set_page_config(page_title="Transformer News Summarizer", page_icon="📝", layout="centered")

# 2. Caching mechanism 
# We cache the model loading using @st.cache_resource so it's only loaded into memory ONCE
# instead of reloading every time a user clicks a button.
@st.cache_resource
def load_model():
    return NewsSummarizer(model_name="facebook/bart-large-cnn")

def main():
    st.title("📰 Abstractive News Summarizer")
    st.markdown("Generate **fully rephrased** abstractive summaries using **Facebook BART-Large-CNN** — a model fine-tuned on news articles to produce concise, multi-sentence summaries without copying the source.")

    # Load Backend Model
    with st.spinner("Loading BART-Large-CNN Model Weights (may take a moment on first run)..."):
        summarizer = load_model()

    # 3. User Input Section
    st.subheader("Input Article")
    article_text = st.text_area(
        "Paste your long-form news article here:", 
        height=250, 
        placeholder="Enter news text here..."
    )
    
    # 4. Configuration Section
    st.subheader("Configuration")
    length_option = st.selectbox(
        "Select Summary Length Target:",
        ("Short", "Medium", "Long"),
        index=1,
        help="Short: 1 sentence, ~20 words. Medium: 2–3 sentences, ~50 words. Long: 4–6 sentences, ~100 words."
    )
    
    # 5. Execution Section
    if st.button("Generate Summary", type="primary"):
        if not article_text.strip():
            st.warning("Please paste some text before generating.")
        else:
            with st.spinner("Generating abstractive summary using Beam Search (num_beams=6)..."):
                try:
                    summary = summarizer.summarize(article_text, length_option=length_option)
                    
                    st.success("Summary Generated!")
                    st.subheader("Output Result:")
                    
                    # Display the final summary inside an informational box
                    st.info(summary)
                    
                except Exception as e:
                    st.error(f"Failed to generate summary: {str(e)}")

if __name__ == "__main__":
    main()
