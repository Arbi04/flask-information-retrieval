from flask import Flask, render_template, request, redirect, url_for
import math
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.createStemmer()

# Data awal (Database sederhana dalam memori)
documents = [
    { 'id': 1, 'text': "Information retrieval adalah proses mencari informasi dari koleksi dokumen" },
    { 'id': 2, 'text': "Cosine similarity mengukur kesamaan antara dua vektor dalam ruang multidimensi" },
    { 'id': 3, 'text': "Sistem temu balik informasi menggunakan berbagai algoritma untuk ranking dokumen" },
    { 'id': 4, 'text': "TF-IDF adalah metode pembobotan kata dalam information retrieval" },
    { 'id': 5, 'text': "Vector space model merepresentasikan dokumen dan query sebagai vektor" }
]

# --- LOGIKA INFORMATION RETRIEVAL ---

def preprocess(text):
    """Tokenisasi, lowercase, hapus tanda baca, dan stemming."""
    text = text.lower()
    # Menghapus karakter non-alphanumeric
    text = re.sub(r'[^\w\s]', '', text)
    # Split berdasarkan spasi dan filter string kosong
    tokens = [word for word in text.split() if word]
    # Stemming setiap token
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def calculate_tf(tokens):
    """Menghitung Term Frequency."""
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    return tf

def calculate_idf(all_docs):
    """Menghitung Inverse Document Frequency."""
    idf = {}
    doc_count = len(all_docs)
    all_terms = set()
    
    # Kumpulkan semua kata unik
    for doc in all_docs:
        tokens = preprocess(doc['text'])
        for token in tokens:
            all_terms.add(token)
    
    # Hitung skor IDF untuk setiap kata
    for term in all_terms:
        docs_with_term = 0
        for doc in all_docs:
            if term in preprocess(doc['text']):
                docs_with_term += 1
        
        # Rumus logaritma (menggunakan basis e seperti Math.log di JS)
        idf[term] = math.log(doc_count / (docs_with_term if docs_with_term > 0 else 1))
        
    return idf

def calculate_tfidf(tokens, idf):
    """Menghitung vektor bobot TF-IDF."""
    tf = calculate_tf(tokens)
    tfidf = {}
    
    for token in tokens:
        tfidf[token] = tf[token] * idf.get(token, 0)
        
    return tfidf

def cosine_similarity(vec1, vec2):
    """Menghitung kesamaan kosinus antara dua vektor."""
    # Gabungkan semua keys dari kedua vektor
    terms = set(vec1.keys()) | set(vec2.keys())
    
    dot_product = 0
    mag1 = 0
    mag2 = 0
    
    for term in terms:
        v1 = vec1.get(term, 0)
        v2 = vec2.get(term, 0)
        
        dot_product += v1 * v2
        mag1 += v1 * v1
        mag2 += v2 * v2
        
    if mag1 == 0 or mag2 == 0:
        return 0
        
    return dot_product / (math.sqrt(mag1) * math.sqrt(mag2))

# --- ROUTES FLASK ---

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""
    
    if request.method == 'POST':
        # Cek apakah user sedang mencari atau menambah dokumen
        if 'search_query' in request.form:
            query = request.form['search_query']
            if query.strip():
                # Proses pencarian
                idf = calculate_idf(documents)
                query_tokens = preprocess(query)
                query_vector = calculate_tfidf(query_tokens, idf)
                
                scored_docs = []
                for doc in documents:
                    doc_tokens = preprocess(doc['text'])
                    doc_vector = calculate_tfidf(doc_tokens, idf)
                    score = cosine_similarity(query_vector, doc_vector)
                    
                    if score > 0:
                        # Salin doc dan tambahkan score
                        d = doc.copy()
                        d['score'] = score
                        scored_docs.append(d)
                
                # Urutkan berdasarkan score tertinggi
                results = sorted(scored_docs, key=lambda x: x['score'], reverse=True)

    return render_template('index.html', documents=documents, results=results, query=query)

@app.route('/add', methods=['POST'])
def add_document():
    text = request.form['new_doc_text']
    if text.strip():
        new_id = max([d['id'] for d in documents], default=0) + 1
        documents.append({'id': new_id, 'text': text})
    return redirect(url_for('index'))

@app.route('/delete/<int:doc_id>')
def delete_document(doc_id):
    global documents
    documents = [d for d in documents if d['id'] != doc_id]
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False)