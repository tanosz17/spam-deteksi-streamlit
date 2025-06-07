from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Data contoh
texts = [
    "Gratis pulsa sekarang juga", 
    "Transfer uang ke saya", 
    "Ayo main bola besok",
    "Diskon besar hari ini", 
    "Undangan rapat siang ini", 
    "Dapat hadiah klik link ini"
]
labels = [1, 1, 0, 1, 0, 1]  # 1: spam, 0: bukan spam

# Training model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# Simpan model
joblib.dump(model, "model_spam.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model dan vectorizer disimpan.")
