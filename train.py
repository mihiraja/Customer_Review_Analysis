import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import re

# Load a sample of the dataset (reduce size for efficiency)
df = pd.read_csv("filtered_reviews.csv")
df = df.sample(n=10000, random_state=42)  # Use only 1000 entries for speed

# Ensure missing values are handled
df['processed_text'] = df['processed_text'].fillna("") 

# Optimized text cleaning function
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
        return text.lower().strip()
    return ""

df['cleaned_text'] = df['processed_text'].apply(clean_text)

# **TF-IDF Vectorizer instead of CountVectorizer**
vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.7, max_features=5000)

# Fit and transform text
X = vectorizer.fit_transform(df['cleaned_text'])

# Prepare target variable (star ratings)
y = df['stars']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Save the model and vectorizer for future use
joblib.dump(model, 'star_rating_model_rf_tfidf.pkl')
joblib.dump(vectorizer, 'vectorizer_tfidf.pkl')

print("Model and vectorizer saved.")
