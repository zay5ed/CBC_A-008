from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load and prepare the dataset
def load_learning_data():
    # Sample CSV structure: topic,description,difficulty,prerequisites
    df = pd.read_csv('learning_data.csv')
    return df

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

@app.route('/generate_learning_path', methods=['POST'])
def generate_learning_path():
    try:
        data = request.get_json()
        interests = data.get('interests', '')
        stream = data.get('stream', '')
        
        # Load the dataset
        df = load_learning_data()
        
        # Combine all text features for similarity comparison
        df['combined_features'] = df['topic'] + ' ' + df['description']
        
        # Create TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
        
        # Convert user interests to TF-IDF
        user_interests_tfidf = vectorizer.transform([interests + ' ' + stream])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_interests_tfidf, tfidf_matrix)
        
        # Get top 5 most relevant topics
        top_indices = similarity_scores[0].argsort()[-5:][::-1]
        
        # Create learning path
        learning_path = []
        for idx in top_indices:
            topic_info = {
                'topic': df.iloc[idx]['topic'],
                'description': df.iloc[idx]['description'],
                'difficulty': df.iloc[idx]['difficulty'],
                'prerequisites': df.iloc[idx]['prerequisites']
            }
            learning_path.append(topic_info)
        
        return jsonify({
            'status': 'success',
            'learning_path': learning_path
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
def home():
    return app.send_static_file('ok - Copy.html')

if __name__ == '__main__':
    app.run(debug=True)
