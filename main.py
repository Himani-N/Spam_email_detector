import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


# Preprocessing function
def preprocess_message(message):
    stop_words = set(stopwords.words('english'))
    words = message.lower().split()  # Split message into words
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

def main():
    # 1️⃣ Load dataset
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].apply(preprocess_message)

    # 2️⃣ Feature extraction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    # 3️⃣ Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4️⃣ Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 5️⃣ Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # 6️⃣ Test with user input (must be here, inside main)
    user_email = input("\nEnter your email/message to check if it's Spam or Not: ")
    user_vector = vectorizer.transform([user_email])
    user_prediction = model.predict(user_vector)

    print("\nResult:")
    print("Spam" if user_prediction[0] == 1 else "Not Spam")


if __name__ == "__main__":
    main()
