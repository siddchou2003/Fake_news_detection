Dataset used
   1. `True.csv`: Real news articles
   2. `Fake.csv`: Fake news articles

Project Structure

fake-news-detector/
├── app.py # Flask app
├── train_model.py
├── model/
│ ├── fake_news_model.pkl
│ └── tfidf_vectorizer.pkl
├── templates/
│ └── index.html
├── static/
│ └── style.css
├── News_dataset/
│ ├── True.csv
│ └── Fake.csv
├── requirements.txt
└── README.md
