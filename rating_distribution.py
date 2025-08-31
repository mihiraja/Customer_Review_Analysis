import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset safely with proper dtypes
df = pd.read_csv("yelp_reviews.csv", dtype={"stars": "float"}, low_memory=False)

# Drop any missing values
df = df.dropna(subset=["stars", "text"])  # Remove rows with missing ratings or text

# Ensure `text` is a string to avoid errors
df["text"] = df["text"].astype(str)

# Add a column for text length
df["text_length"] = df["text"].apply(lambda x: len(x.split()))


# Distribution of ratings
plt.figure(figsize=(8, 5))
sns.countplot(x=df["stars"], palette="viridis")
plt.title("Distribution of Yelp Ratings")
plt.xlabel("Stars")
plt.ylabel("Count")
plt.show()
