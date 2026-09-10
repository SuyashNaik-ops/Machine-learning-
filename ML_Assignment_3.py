import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv(r"C:\Users\PRAJWAL\Downloads\dattaset\IMDB Dataset.csv")
df = df.dropna()

print("Dataset Shape:", df.shape)
print(df.head())

X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

reviews = [
    "This movie was excellent and I really enjoyed it.",
    "The movie was boring and a complete waste of time."
]

reviews_tfidf = vectorizer.transform(reviews)
predictions = model.predict(reviews_tfidf)

print("\nCustom Review Predictions:")
for review, prediction in zip(reviews, predictions):
    print("\nReview:", review)
    print("Sentiment:", prediction)