from flask import Flask, render_template, request, redirect, url_for
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ===============================
# INISIALISASI STEMMER & STOPWORD
# ===============================
stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())

# ===============================
# DATA DOKUMEN (IN-MEMORY)
# ===============================
documents = [
    {'id': 1, 'text': "Information retrieval adalah proses mencari informasi dari koleksi dokumen"},
    {'id': 2, 'text': "Cosine similarity mengukur kesamaan antara dua vektor dalam ruang multidimensi"},
    {'id': 3, 'text': "Sistem temu balik informasi menggunakan berbagai algoritma untuk ranking dokumen"},
    {'id': 4, 'text': "TF-IDF adalah metode pembobotan kata dalam information retrieval"},
    {'id': 5, 'text': "Vector space model merepresentasikan dokumen dan query sebagai vektor"}
]

# ===============================
# PREPROCESSING TEKS
# ===============================
def preprocess(text):
    """
    Tahapan preprocessing:
    1. Lowercase
    2. Hapus tanda baca
    3. Tokenisasi
    4. Stopword removal
    5. Stemming Bahasa Indonesia
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

# ===============================
# TF-IDF PRECOMPUTATION
# ===============================
def build_tfidf_model():
    """
    Menghitung TF-IDF dokumen SATU KALI
    """
    corpus = [preprocess(doc['text']) for doc in documents]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return vectorizer, tfidf_matrix

# Inisialisasi awal
vectorizer, tfidf_matrix = build_tfidf_model()

# ===============================
# ROUTE INDEX + SEARCH
# ===============================
@app.route('/', methods=['GET', 'POST'])
def index():
    global vectorizer, tfidf_matrix

    results = []
    query = ""

    if request.method == 'POST' and 'search_query' in request.form:
        query = request.form['search_query']

        if query.strip():
            processed_query = preprocess(query)

            # TF-IDF QUERY (TANPA FIT ULANG)
            query_vector = vectorizer.transform([processed_query])

            # COSINE SIMILARITY
            similarity_scores = cosine_similarity(
                query_vector,
                tfidf_matrix
            )[0]

            # RANKING DOKUMEN
            for idx, score in enumerate(similarity_scores):
                if score > 0:
                    doc = documents[idx].copy()
                    doc['score'] = float(score)
                    results.append(doc)

            results.sort(key=lambda x: x['score'], reverse=True)

    return render_template(
        'index.html',
        documents=documents,
        results=results,
        query=query
    )

# ===============================
# ROUTE TAMBAH DOKUMEN
# ===============================
@app.route('/add', methods=['POST'])
def add_document():
    global vectorizer, tfidf_matrix

    text = request.form['new_doc_text']
    if text.strip():
        new_id = max([d['id'] for d in documents], default=0) + 1
        documents.append({'id': new_id, 'text': text})

        # REBUILD TF-IDF SAAT DOKUMEN BERUBAH
        vectorizer, tfidf_matrix = build_tfidf_model()

    return redirect(url_for('index'))

# ===============================
# ROUTE HAPUS DOKUMEN
# ===============================
@app.route('/delete/<int:doc_id>')
def delete_document(doc_id):
    global documents, vectorizer, tfidf_matrix

    documents = [d for d in documents if d['id'] != doc_id]

    # REBUILD TF-IDF SAAT DOKUMEN BERUBAH
    vectorizer, tfidf_matrix = build_tfidf_model()

    return redirect(url_for('index'))

# ===============================
# MAIN
# ===============================
if __name__ == '__main__':
    app.run(debug=False)