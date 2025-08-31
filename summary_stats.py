import pandas as pd


# Load dataset safely with proper dtypes
df = pd.read_csv("yelp_reviews.csv", dtype={"stars": "float"}, low_memory=False)

# Drop any missing values
df = df.dropna(subset=["stars", "text"])  # Remove rows with missing ratings or text

# Ensure `text` is a string to avoid errors
df["text"] = df["text"].astype(str)

# Add a column for text length
df["text_length"] = df["text"].apply(lambda x: len(x.split()))

# Summary statistics for review length
print(df["text_length"].describe())

print(df["stars"].describe())