# %%
import pandas as pd
books=pd.read_csv("books.csv")
print(books)
# %%
import seaborn as sns
import matplotlib.pyplot as plt
ax=plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)
plt.xlabel("Columns")
plt.ylabel("Missing values")
plt.show()
# %%
import numpy as np
books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2024 - books["published_year"]
columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]
correlation_matrix = books[columns_of_interest].corr(method = "spearman")
sns.set_theme(style="white")
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                      cbar_kws={"label": "Spearman correlation"})
heatmap.set_title("Correlation heatmap")
plt.show()
# %%
book_missing = books[~(books["description"].isna()) &
      ~(books["num_pages"].isna()) &
      ~(books["average_rating"].isna()) &
      ~(books["published_year"].isna())
]
book_missing
# %%
book_missing["categories"].value_counts().reset_index().sort_values("count", ascending=False)
# %%
book_missing
# %%
book_missing["words_in_description"] = book_missing["description"].str.split().str.len()
book_missing
# %%
book_missing.loc[book_missing["words_in_description"].between(1, 4), "description"]
# %%
book_missing.loc[book_missing["words_in_description"].between(5, 14), "description"]
# %%
book_missing.loc[book_missing["words_in_description"].between(15, 24), "description"]
# %%
book_missing.loc[book_missing["words_in_description"].between(25, 34), "description"]
# %%
book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25]
book_missing_25_words
# %%
book_missing_25_words["title_and_subtitle"] = (
    np.where(book_missing_25_words["subtitle"].isna(), book_missing_25_words["title"],
             book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1))
)
book_missing_25_words
# %%
book_missing_25_words["tagged_description"] = book_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
book_missing_25_words
# %%
(
    book_missing_25_words
    .drop(["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1)
    .to_csv("books_cleaned.csv", index = False)
)
# %%
