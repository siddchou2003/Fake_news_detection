import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Load and label data
true_df = pd.read_csv("News_dataset/True.csv")
fake_df = pd.read_csv("News_dataset/Fake.csv")
true_df['label'] = 1
fake_df['label'] = 0

# Optional: balance dataset
min_len = min(len(true_df), len(fake_df))
true_df = true_df.sample(min_len, random_state=42)
fake_df = fake_df.sample(min_len, random_state=42)

# Combine and clean
data = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
data['cleaned'] = data['text'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
X = vectorizer.fit_transform(data['cleaned'])
y = data['label']

# Train model
model = PassiveAggressiveClassifier()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')