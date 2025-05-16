# Tweet Sentiment Analysis Application

A Flask web application for analyzing sentiment in tweets using machine learning.

## Requirements
- Python 3.8+
- pip

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Koleabhi/tweet_sentiment_app.git
cd tweet_sentiment_app
Install dependencies:

bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet punkt_tab
Place model files in model/ directory:

logistic_regression_model.pkl

tfidf_vectorizer.pkl

Usage
bash
python app.py
Visit http://localhost:5000 in your browser

Dataset
The model was trained on the Sentiment140 dataset:

Download link: [http://help.sentiment140.com/for-students](https://www.kaggle.com/datasets/kazanova/sentiment140)

Format: 1.6 million tweets with sentiment labels (0 = negative, 4 = positive)

Note: You'll need to preprocess the dataset and retrain the model using the data_modeling.ipynb notebook if you want to use custom training data.




