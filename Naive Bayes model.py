import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load Data
df = pd.read_csv("train.csv", encoding='latin1') 
df = df.dropna(subset=['text', 'sentiment']) # Replace with your actual CSV file name

# STEP 2: Feature Extraction
X = df['text']
y = df['sentiment']

vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

# STEP 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split( X_vect,y, test_size=0.3, random_state=42)

# STEP 4: Train Naive Bayes Model
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

# STEP 5: Predict using Naive Bayes
y_pred_nb = model_nb.predict(X_test)

# STEP 6: Evaluate Naive Bayes
print("🔍 Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("✅ Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb)*100, "%")

# STEP 7: Predict on full dataset for comparison
df['NaiveBayes Sentiment'] = model_nb.predict(X_vect)
comparison_df = pd.DataFrame({
    'Text': X_test,
    'Actual Sentiment': y_test,
    'Predicted Sentiment': y_pred_nb
}).reset_index(drop=True)

print("\nActual vs Predicted:\n")
print(comparison_df.to_string(index=False))
comparison_df.to_excel("Actual vs Predicted Sentiment for Naive Bayes.xlsx",index = False)
accuracy = accuracy_score(y_test, y_pred_nb)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
actual_counts = comparison_df['Actual Sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Actual Count')
predicted_counts = comparison_df['Predicted Sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Predicted Count')

# Merge to compare side by side
sentiment_comparison = pd.merge(actual_counts, predicted_counts, on='Sentiment', how='outer').fillna(0)

print(sentiment_comparison)
sentiment_comparison.to_excel("Count of sentiment for NB.xlsx", index = False)
import matplotlib.pyplot as plt
sentiment_comparison.plot(kind='bar', figsize=(8,5), title='Sentiment Distribution: Actual vs Predicted Naive Bayes classification')
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
#### Thank You ####