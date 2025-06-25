# %%
#Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
#Viewing data
books=pd.read_csv("../../data/raw/books.csv")
print(books)

# %%
#Finding what kind of data present
df = pd.DataFrame(books) 
df.info()

# %%
#Checking missing values of each column
missing_values = df.isna().sum()
print("Missing values per column:")
print(missing_values)

# %%
#Checking distinct values in each column
distinct_values = df.nunique()
print("\nDistinct values per column:")
print(distinct_values)

# %%
#Full summary
summary = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isna().sum(),
    'Total Values': df.count(),
    'Unique Values': df.nunique()
})
print("\nComplete Summary:")
print(summary)

# %%
#Checking connections between missing values using heatmap
ax=plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)
plt.xlabel("Columns")
plt.ylabel("Missing values")
plt.show()

# %%
#Checking relationship between missing values
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
#Fetching values with missing fields 
book_missing = books[(books["description"].isna()) |
      (books["num_pages"].isna()) |
      (books["average_rating"].isna()) |
      (books["published_year"].isna())
]
book_missing #303 rows

# %%
#Removing missing fields
book_missing = books[~(books["description"].isna()) &
      ~(books["num_pages"].isna()) &
      ~(books["average_rating"].isna()) &
      ~(books["published_year"].isna())
]
book_missing #6507 rows

# %%
#Checking distribution of categories column
book_missing["categories"].value_counts().reset_index().sort_values("count", ascending=False)

# %%
#Checking total words in description
book_missing["words_in_description"] = book_missing["description"].str.split().str.len()
book_missing

# %%
#Checking a 25 word description
book_missing.loc[book_missing["words_in_description"].between(25, 34), "description"]

# %%
#Removed Books with < 25  words in description
book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25]
book_missing_25_words #5197

# %%
#Created new column for merging title and sub-title
book_missing_25_words["title_and_subtitle"] = (
    np.where(book_missing_25_words["subtitle"].isna(), book_missing_25_words["title"],
             book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1))
)
book_missing_25_words

# %%
#Created a column with isbn and description
book_missing_25_words["tagged_description"] = book_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
book_missing_25_words

# %%
#Removed usless column and saved it to new csv file
(
    book_missing_25_words
    .drop(["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1)
    .to_csv("../../data/processed/books_cleaned.csv", index = False)
)
# %%
