import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define stop words
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)  # Remove URLs and mentions
    text = re.sub(r"[^a-z\s]", "", text)  # Remove punctuation and numbers

    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Basic negation handling
    negation_words = ["not", "never", "no"]
    negated = False
    processed_words = []

    for word in words:
        if word in negation_words:
            negated = True
        elif negated:
            processed_words.append("not_" + word)
            negated = False
        else:
            processed_words.append(lemmatizer.lemmatize(word))  # Lemmatize words

    return " ".join(processed_words)

# Load your data
df = pd.read_csv("C:\PROJECTS\Sentiment Analysis\emotion_sentiment_analysis.py\sentiment_emotion_dataset.csv")

# Data Exploration
print("Dataframe Head:\n", df.head())
print("\nSentiment Distribution:\n", df['sentiment'].value_counts())
print("\nEmotion Distribution:\n", df['emotion'].value_counts())

# Preprocess text
df['text'] = df['text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train_sentiment, y_test_sentiment = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

X_train, X_test, y_train_emotion, y_test_emotion = train_test_split(
    df['text'], df['emotion'], test_size=0.2, random_state=42
)

# Define sentiment model pipeline
sentiment_model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))  # Added max_iter
])

# Define emotion model pipeline
emotion_model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))  # Added max_iter
])

# Parameter grid for sentiment model tuning
sentiment_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': [True, False],
    'clf__C': [0.1, 1, 10]
}

# Parameter grid for emotion model tuning
emotion_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': [True, False],
    'clf__C': [0.1, 1, 10]
}

# Grid search for sentiment model
sentiment_grid_search = GridSearchCV(sentiment_model, sentiment_param_grid, cv=3, scoring='accuracy')
sentiment_grid_search.fit(X_train, y_train_sentiment)
best_sentiment_model = sentiment_grid_search.best_estimator_

# Grid search for emotion model
emotion_grid_search = GridSearchCV(emotion_model, emotion_param_grid, cv=3, scoring='accuracy')
emotion_grid_search.fit(X_train, y_train_emotion)
best_emotion_model = emotion_grid_search.best_estimator_

# Predict using the best models
sentiment_pred = best_sentiment_model.predict(X_test)
emotion_pred = best_emotion_model.predict(X_test)

# Evaluate models
print("Sentiment Model Accuracy:", accuracy_score(y_test_sentiment, sentiment_pred))
print("Emotion Model Accuracy:", accuracy_score(y_test_emotion, emotion_pred))

print("\nSentiment Model Classification Report:\n", classification_report(y_test_sentiment, sentiment_pred))
print("\nEmotion Model Classification Report:\n", classification_report(y_test_emotion, emotion_pred))

# Interactive loop
while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    processed_input = preprocess_text(user_input)
    print("Processed Input:", processed_input) #Debugging
    sentiment = best_sentiment_model.predict([processed_input])[0]
    emotion = best_emotion_model.predict([processed_input])[0]

    print("Text:", user_input)
    print("Predicted Sentiment:", sentiment)
    print("Predicted Emotion:", emotion)