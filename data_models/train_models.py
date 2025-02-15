import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("passwords.csv").dropna()

# Convert passwords to numerical features
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
X = vectorizer.fit_transform(df["password"])
y = df["strength"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model & vectorizer
joblib.dump(clf, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

