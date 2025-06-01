# Sentiment Analysis using CSV file input

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load CSV file
# Make sure your file has columns named 'text' and 'sentiment'
df = pd.read_csv("train.csv", encoding = 'latin1')  # Replace with your actual file name
df = df.dropna(subset=['text', 'sentiment'])
# Step 2: Display basic info
print("Sample Data:")
print(df.head())

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.3, random_state=42
)

# Step 4: Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Step 5: Train model
pipeline.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Step 7: Accuracy


# Step 8: Actual vs Predicted Comparison
comparison_df = pd.DataFrame({
    'Text': X_test.values,
    'Actual Sentiment': y_test.values,
    'Predicted Sentiment': y_pred
}).reset_index(drop=True)

print("\nActual vs Predicted:\n")
print(comparison_df.to_string(index=False))
comparison_df.to_excel("Actual vs Predicted Sentiment.xlsx",index = False)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
actual_counts = comparison_df['Actual Sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Actual Count')
predicted_counts = comparison_df['Predicted Sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Predicted Count')

# Merge to compare side by side
sentiment_comparison = pd.merge(actual_counts, predicted_counts, on='Sentiment', how='outer').fillna(0)

print(sentiment_comparison)
sentiment_comparison.to_excel("Count of sentiment.xlsx", index = False)
import matplotlib.pyplot as plt
sentiment_comparison.plot(kind='bar', figsize=(8,5), title='Sentiment Distribution: Actual vs Predicted')
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
#### Thank You ####