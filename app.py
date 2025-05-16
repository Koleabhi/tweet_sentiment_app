from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib

app = Flask(__name__)

# Load models with error handling
try:
    model = joblib.load('model/logistic_regression_model.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
except Exception as e:
    model = None
    vectorizer = None
    print(f"Error loading models: {e}")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Clean text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Keep only letters
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model is not None)

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['tweet']
        
        if not text.strip():
            return render_template('index.html', 
                                 error="Please enter some text to analyze",
                                 model_loaded=model is not None)
        
        if not model or not vectorizer:
            return render_template('index.html', 
                                 error="Model failed to load. Please check server logs",
                                 model_loaded=False)
        
        # Preprocess and predict
        cleaned_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)
        result = "Positive" if prediction[0] == 1 else "Negative"
        
        return render_template('index.html', 
                             result=result,
                             original_text=text,
                             model_loaded=True)
    
    return render_template('index.html', model_loaded=model is not None)
print(f"Model loaded: {model is not None}")
print(f"Vectorizer loaded: {vectorizer is not None}")

if __name__ == '__main__':
    app.run(debug=True)