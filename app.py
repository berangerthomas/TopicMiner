import os
import re
import tempfile

import nltk

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

import pymupdf4llm
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# Prétraitement du texte Markdown
def preprocess_markdown(markdown_text):
    # Supprimer la syntaxe Markdown
    text = re.sub(r"#|\*|_|\[.*?\]|\(.*?\)|`.*?`", "", markdown_text)

    # Tokenisation et nettoyage
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words("french"))  # ou 'english' selon votre langue
    lemmatizer = WordNetLemmatizer()

    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]

    return " ".join(processed_tokens)


def main():
    st.title("Topic Miner")

    # Add description
    st.write("Upload a PDF file to identify the main topics in the document.")

    # File uploader widget with drag and drop capability
    uploaded_file = st.file_uploader("Drag and drop a PDF file", type=["pdf"])

    md_text = None

    if uploaded_file is not None:
        # Display success message
        st.success("File successfully uploaded!")

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        try:
            # Process the PDF using pymupdf4llm
            # Show processing indicator
            with st.spinner("Converting PDF to markdown..."):
                md_text = pymupdf4llm.to_markdown(pdf_path)
                md_text = md_text.encode("utf-8", errors="replace").decode("utf-8")
                st.success("Conversion complete!")

            # Display the markdown content
            st.subheader("Generated Markdown Content (first 20 lines):")
            # Limit display to first 20 lines
            md_lines = md_text.split("\n")
            if len(md_lines) > 20:
                md_text_display = "\n".join(md_lines[:20])
                st.markdown(md_text_display)
                st.info(f"Showing only first 20 lines of {len(md_lines)} total lines.")
            else:
                st.markdown(md_text)

            # Add download button for the complete markdown file
            # Get the original filename and replace extension
            original_filename = uploaded_file.name
            md_filename = os.path.splitext(original_filename)[0] + ".md"

            # Store file info in session state to avoid reprocessing
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = {}

            file_key = f"{original_filename}_{uploaded_file.size}"
            st.session_state.processed_files[file_key] = md_text

            st.download_button(
                label="Download full markdown file",
                # data=md_text.encode("utf-8"),
                data=md_text,
                file_name=md_filename,
                mime="text/markdown",
                key=f"download_{file_key}",  # Unique key prevents widget recreation
            )

        except Exception as e:
            st.error(f"Error processing PDF: {e}")

        try:
            # Extracting main topics using LDA from scikit-learn
            with st.spinner(
                "Extracting main topics using Latent Dirichlet Allocation..."
            ):
                # Diviser le texte en paragraphes ou sections pour créer un corpus
                paragraphs = re.split(r"\n\n+", md_text)
                processed_paragraphs = [
                    preprocess_markdown(p) for p in paragraphs if p.strip()
                ]

                # Vectorisation
                count_vectorizer = CountVectorizer(max_features=1000)
                count_data = count_vectorizer.fit_transform(processed_paragraphs)

                # Application de LDA
                lda = LatentDirichletAllocation(n_components=5, random_state=0)
                lda.fit(count_data)

                # Extract top words for each topic
                feature_names = count_vectorizer.get_feature_names_out()
                n_top_words = 2
                topics = []

                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[: -n_top_words - 1 : -1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

                st.success("Topics extracted!")

            # Display the extracted topics
            st.subheader("Main Topics:")
            for topic in topics:
                st.write(f"- {topic}")

        except Exception as e:
            st.error(f"Error extracting topics: {e}")

        finally:
            # Remove the temporary file
            os.unlink(pdf_path)

    return md_text


if __name__ == "__main__":
    markdown_variable = main()
