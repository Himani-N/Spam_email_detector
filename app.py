from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Preprocess messages
def preprocess_message(message):
    stop_words = set(stopwords.words('english'))
    words = message.lower().split()
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

# Initialize Flask
app = Flask(__name__)

# Load dataset and train model
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1','v2']]
df.columns = ['label','message']
df['label'] = df['label'].map({'ham':0,'spam':1})
df['message'] = df['message'].apply(preprocess_message)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Route for home page
@app.route("/", methods=["GET","POST"])
def index():
    result = ""
    if request.method == "POST":
        email_text = request.form["email_content"]
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
