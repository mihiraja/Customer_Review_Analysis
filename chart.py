import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz

# Load the sparse matrix correctly
X = load_npz("bow_reviews_sparse.npz")
print("Sparse matrix shape:", X.shape)  # Check if the matrix is loaded properly

# Load vocabulary from CSV
vocab_df = pd.read_csv("bow_vocab.csv")
vocab = vocab_df["word"].tolist()  # Convert to a list of words

# Sum occurrences across all documents
word_counts = np.asarray(X.sum(axis=0)).flatten()

# Take the top 20 most frequent words
top_n = 20
top_indices = word_counts.argsort()[-top_n:][::-1]  # Get indices of top N words

# Filter vocabulary for the top words
top_words = [vocab[i] for i in top_indices]
top_counts = word_counts[top_indices]

# Print the top words and their counts
print(list(zip(top_words, top_counts)))  # Print the top words and their counts

# Create a DataFrame for visualization
top_words_df = pd.DataFrame({"word": top_words, "count": top_counts})

# Plot bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x="count", y="word", data=top_words_df, palette="viridis")
plt.xlabel("Word Frequency")
plt.ylabel("Words")
plt.title("Top 20 Most Frequent Words in Bag-of-Words Representation")
plt.show()


