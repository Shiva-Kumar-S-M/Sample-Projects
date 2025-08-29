# model_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib  # To save the model

# 1. Create a simple dataset. For a real project, you'd use a larger one (e.g., from Kaggle).
# Format: [text, sentiment]
data = [
    ["I love this product, it's amazing!", "positive"],
    ["This is the worst thing I have ever bought.", "negative"],
    ["It's okay, nothing special.", "neutral"],
    ["The package arrived early and I'm thrilled.", "positive"],
    ["Absolutely terrible quality, broke immediately.", "negative"],
    ["It's a standard item, does the job.", "neutral"],
    # ... Add many more examples! (~100-200 of each sentiment for a decent model)
]

# Convert to a DataFrame
df = pd.DataFrame(data, columns=['text', 'sentiment'])

# 2. Split into Features (X) and Target (y)
X = df['text']
y = df['sentiment']

# 3. Create a pipeline that does two things:
#   a. Vectorize: Convert text to TF-IDF numbers
#   b. Classify: Use a Logistic Regression model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# 4. Train the model
model.fit(X, y)

# 5. Save the trained model to a file so we can use it later without retraining
joblib.dump(model, 'sentiment_model.pkl')
print("Model trained and saved successfully!")