import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz
import re

# Load the dataset (use a sample to reduce size if needed)
df = pd.read_csv("filtered_reviews.csv")
df = df.sample(frac=0.3, random_state=42)  # Take a 30% sample

# Ensure missing values are replaced
df['processed_text'] = df['processed_text'].fillna("")

# Optimized text cleaning function
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
        return text.lower().strip()
    return ""

df['cleaned_text'] = df['processed_text'].apply(clean_text)

# Optimized CountVectorizer
vectorizer = CountVectorizer(stop_words='english', min_df=5, max_df=0.7, max_features=10_000)

# Fit and transform text
X = vectorizer.fit_transform(df['cleaned_text'])

# Save sparse matrix
save_npz("bow_reviews_sparse.npz", X, compressed=True)

# Save vocabulary
pd.DataFrame(vectorizer.get_feature_names_out(), columns=["word"]).to_csv("bow_vocab.csv", index=False)

print("Optimized Bag-of-Words matrix and vocabulary saved.")
