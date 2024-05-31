from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

app = Flask(__name__)

# Load the Universal Sentence Encoder model
use_model = None

def load_model():
    global use_model
    try:
        use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    except Exception as e:
        print("Error loading model: ", e)

# Chunk text into 512 token chunks
def chunk_text(text, chunk_size=512):
    words = text.split(" ")
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Encode text chunks
def encode_text(text):
    chunks = chunk_text(text)
    embeddings = use_model(chunks)
    return embeddings

# Compute similarity scores
def compute_similarity_scores(doc_embeddings, abstract_embeddings):
    scores = []
    for doc_embedding in doc_embeddings:
        max_score = 0
        for abstract_embedding in abstract_embeddings:
            # Using tf.keras.losses.cosine_similarity directly
            score = 1 - tf.keras.losses.cosine_similarity(doc_embedding, abstract_embedding).numpy()
            max_score = max(max_score, score)
        scores.append(max_score)
    return scores

# Rank abstracts based on similarity scores
def rank_abstracts(abstracts, scores):
    ranked = sorted(
        [{"PMID": abstract["PMID"], "score": score}
         for abstract, score in zip(abstracts, scores)],
        key=lambda x: x["score"],
        reverse=True
    )
    return ranked

@app.route('/compare_abstracts', methods=['POST'])
def compare_abstracts():
    data = request.json
    document = data['document']
    abstracts = data['abstracts']

    if use_model is None:
        load_model()
        if use_model is None:
            return jsonify({'error': 'Model loading error'}), 500

    # Encode document chunks
    doc_embeddings = encode_text(document)

    # Encode abstract chunks
    abstract_embeddings = [encode_text(abstract["Abstract"]) for abstract in abstracts]

    # Compute similarity scores for each abstract
    scores = [compute_similarity_scores(doc_embeddings, embeddings) for embeddings in abstract_embeddings]

    # Average the scores for each abstract
    average_scores = [sum(score_array) / len(score_array) for score_array in scores]

    # Rank abstracts
    ranked_abstracts = rank_abstracts(abstracts, average_scores)

    # Create sorted list of PMID numbers with their positions
    sorted_pmids = [{"PMID": item["PMID"], "position": idx + 1} for idx, item in enumerate(ranked_abstracts)]
    
    return jsonify(sorted_pmids)

if __name__ == '__main__':
    app.run(debug=True)
