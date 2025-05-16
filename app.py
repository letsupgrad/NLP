import streamlit as st
import pandas as pd
import pdfplumber
import docx
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from transformers import pipeline
import spacy
from langdetect import detect
from googletrans import Translator
import whisper
import tempfile
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit.components.v1 as components

# Initialize downloads for nltk if needed
nltk.download("punkt")
nltk.download("stopwords")

# --- Load models with caching ---
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    nlp = spacy.load("en_core_web_sm")
    whisper_model = whisper.load_model("base")
    return summarizer, nlp, whisper_model

summarizer, ner_model, whisper_model = load_models()

translator = Translator()

# --- Utility functions ---
def extract_text_from_file(file, filetype):
    if filetype == "txt":
        return file.read().decode("utf-8")
    elif filetype == "pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif filetype == "docx":
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif filetype == "csv":
        df = pd.read_csv(file)
        return df.to_string()
    else:
        return ""

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.text for p in paragraphs)
    except Exception as e:
        return f"Error fetching URL: {e}"

def summarize_text(text, num_sentences=5):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [w for w in words if w.isalnum() and w not in stop_words]

    freq = pd.Series(filtered_words).value_counts()
    sentence_scores = {}
    for sent in sentences:
        score = sum(freq.get(word.lower(), 0) for word in word_tokenize(sent))
        sentence_scores[sent] = score

    ranked = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = " ".join(ranked[:num_sentences])
    return summary

def get_keywords(text, top_n=20):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum()]
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in words if w not in stop_words]
    freq = pd.Series(filtered).value_counts().reset_index()
    freq.columns = ["Keyword", "Count"]
    return freq.head(top_n)

def plot_wordcloud(freq_df):
    word_freq = dict(zip(freq_df["Keyword"], freq_df["Count"]))
    wc = WordCloud(width=800, height=300, background_color="white").generate_from_frequencies(word_freq)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang != "en":
            translated = translator.translate(text, src=detected_lang, dest="en")
            return translated.text, detected_lang
        return text, "en"
    except:
        return text, "en"

def translate_back(text, lang):
    try:
        if lang != "en":
            return translator.translate(text, src="en", dest=lang).text
        return text
    except:
        return text

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        result = whisper_model.transcribe(tmp.name)
        return result["text"]

def show_lda(text):
    words = [word for word in word_tokenize(text.lower()) if word.isalnum() and word not in stopwords.words("english")]
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    html = pyLDAvis.prepared_data_to_html(vis)
    components.html(html, width=1000, height=600)

# --- Streamlit UI ---
st.set_page_config("Ultimate NLP Summarizer", layout="wide")
st.title("🧠 Ultimate AI Text Summarizer & Keyword Extractor")

# Sidebar
st.sidebar.header("Input & Settings")
input_method = st.sidebar.radio("Choose Input Method:", ["📁 Upload File", "🔗 Enter URL", "🎤 Upload Audio"])
num_sentences = st.sidebar.slider("Summary Length (sentences)", 1, 15, 5)
summarization_type = st.sidebar.radio("Summarization Type:", ["Extractive (TF-IDF)", "Abstractive (T5)"])
show_topics = st.sidebar.checkbox("Show Topics (LDA)", value=False)

text = ""
filename = None

# Input Section
if input_method == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload a file (txt, pdf, docx, csv)", type=["txt", "pdf", "docx", "csv"])
    if uploaded_file:
        filetype = uploaded_file.name.split(".")[-1].lower()
        filename = uploaded_file.name
        text = extract_text_from_file(uploaded_file, filetype)

elif input_method == "🔗 Enter URL":
    url = st.text_input("Enter URL:")
    if url:
        text = extract_text_from_url(url)
        filename = "webpage"

elif input_method == "🎤 Upload Audio":
    audio_file = st.file_uploader("Upload audio (mp3, wav)", type=["mp3", "wav"])
    if audio_file:
        st.info("Transcribing audio...")
        text = transcribe_audio(audio_file)
        filename = "audio_transcript"
        st.write("📝 Transcript:")
        st.write(text)

# Preview
if text:
    st.subheader("👀 Preview of Input Text (first 1000 chars)")
    st.text(text[:1000] + ("..." if len(text) > 1000 else ""))

    # Translate if needed
    text_en, orig_lang = translate_to_english(text)

    # Summarize
    if st.button("🔍 Generate Summary"):
        with st.spinner("Summarizing..."):
            if summarization_type == "Abstractive (T5)":
                short_text = text_en[:1000]
                result = summarizer(short_text, max_length=150, min_length=40, do_sample=False)
                summary = result[0]["summary_text"]
            else:
                summary = summarize_text(text_en, num_sentences)

            summary = translate_back(summary, orig_lang)

        st.subheader("🧠 Summary")
        st.success(summary)

        # Download button
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name=f"{filename}_summary.txt" if filename else "summary.txt",
            mime="text/plain",
        )

        # Keywords and Wordcloud
        st.subheader("🔑 Keywords")
        keywords = get_keywords(text_en)
        col1, col2 = st.columns(2)
        with col1:
            st.table(keywords)
        with col2:
            plot_wordcloud(keywords)

        # Named Entities
        st.subheader("🧬 Named Entities")
        doc = ner_model(text_en[:2000])
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            df_entities = pd.DataFrame(entities, columns=["Entity", "Type"])
            st.table(df_entities)
        else:
            st.info("No named entities found.")

        # Topic modeling
        if show_topics:
            st.subheader("📊 Topic Modeling (LDA)")
            show_lda(text_en)

else:
    st.info("Upload a file, enter a URL, or upload audio to get started!")
