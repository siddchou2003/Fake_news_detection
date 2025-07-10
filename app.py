from flask import Flask, render_template, request
import joblib
import string
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        news_text = request.form['news']
        cleaned = clean_text(news_text)  # 🧼 Clean input
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)