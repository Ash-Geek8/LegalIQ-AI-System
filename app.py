from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import pandas as pd
import json
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer
import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BartForConditionalGeneration, BartTokenizer
import PyPDF2
import docx
from collections import Counter
import re

app = Flask(__name__)
app.secret_key = 'legal-iq'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
tokenizer_s = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Allowed file types
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

sec_cont,sec_emb = None, None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath):
    """Extracts text from PDF or DOCX files."""
    text = ""
    if filepath.endswith('.pdf'):
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = '\n'.join(page.extract_text() for page in reader.pages if page.extract_text())
    elif filepath.endswith('.docx'):
        doc = docx.Document(filepath)
        text = '\n'.join(para.text for para in doc.paragraphs)
    return text.strip()

def summarize_text(text):
    """Summarizes the given document text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs['input_ids'], max_length=200, min_length=50, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_keywords(text, top_n=5):
    """Extracts top keywords to infer the document's purpose."""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    common_words = Counter(words).most_common(top_n)
    return [word for word, _ in common_words]

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)


def retrieve_legal_info(query, vectorizer, legal_text_vectors, df):
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, legal_text_vectors).flatten()
    best_match_idx = np.argmax(similarity_scores)
    return {
        "Matched_Legal_Text": df.loc[best_match_idx, "content"],
        "Relevance_Score": similarity_scores[best_match_idx]
    }

def get_legal_answer(query, context):
    result = qa_pipeline({'context': context, 'question': query})
    return result['answer']

# Past Cases route
def load_data():
    return pd.read_csv("legal_dataset.csv")

def train_vectorizer(texts):
    vectorizer = TfidfVectorizer()
    legal_text_vectors = vectorizer.fit_transform(texts)
    return vectorizer, legal_text_vectors

# Load data and initialize models
df = load_data()
vectorizer, legal_text_vectors = train_vectorizer(df["content"])


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/past_cases', methods=['GET', 'POST'])
def past_cases():
    query = None
    text = None
    summary = None
    answer = None
    if request.method == 'POST':
        if 'query' in request.form and 'summary' not in request.form:
            query = request.form['query']
            retrieved_result = retrieve_legal_info(query, vectorizer, legal_text_vectors, df)
            text = retrieved_result["Matched_Legal_Text"]
            summary = summarize_text(text)
        elif 'query' in request.form and 'summary' in request.form:
            query = request.form['query']
            summary = request.form['summary']
            text = request.form['text']
            answer = get_legal_answer(query, summary)
    
    return render_template('past_cases.html', query=query, text=text, summary=summary, answer=answer)

@app.route('/document_analyzer', methods=['GET', 'POST'])
def document_analyzer():
    global sec_cont,sec_emb
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract document content
            content = extract_text(filepath)
            summary = summarize_text(content)
            keywords = extract_keywords(content)

            # Compute document embeddings
            doc_embedding = embedding_model.encode(content, convert_to_tensor=True).tolist()

            # Store in session
            session['document_content'] = content
            sec_cont = content
            sec_emb = doc_embedding
            session['document_summary'] = summary
            session['document_keywords'] = keywords
            session['document_embeddings'] = doc_embedding

            return render_template('document_analyzer.html', content=content, summary=summary, keywords=keywords)
    return render_template('document_analyzer.html')

@app.route('/chat', methods=['POST'])
def chat():
    global sec_cont,sec_emb
    data = request.get_json()
    user_query = data.get('query', '')

    #if 'document_content' not in session:
    #    return jsonify({'response': "No document uploaded. Please upload a document first."})

    document_text = sec_cont #session['document_content']
    doc_embedding = torch.tensor(sec_emb )#session['document_embeddings'])

    # Compute similarity between query and document
    user_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(user_embedding, doc_embedding)[0].item()

    # Generate response if similarity is high
    if similarity > 0:
        input_text = f"Context: {document_text}\nQuestion: {user_query}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = summarization_model.generate(**inputs, max_length=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        response = "I couldn't find relevant information in the document."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
