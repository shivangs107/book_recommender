# %%
#Importing libraries
import pandas as pd
from transformers import pipeline
import numpy as np
from tqdm import tqdm #For progress bar as classification will take time

# %%
#Loading clean data
books = pd.read_csv("data/processed/books_cleaned.csv")

# %%
#Categories having > 50 books 
books["categories"].value_counts().reset_index().query("count > 50")

# %%
#Categorising as fiction or non-fiction and added children's category too
category_mapping = {'Fiction' : "Fiction",
 'Juvenile Fiction': "Children's Fiction",
 'Biography & Autobiography': "Nonfiction",
 'History': "Nonfiction",
 'Literary Criticism': "Nonfiction",
 'Philosophy': "Nonfiction",
 'Religion': "Nonfiction",
 'Comics & Graphic Novels': "Fiction",
 'Drama': "Fiction",
 'Juvenile Nonfiction': "Children's Nonfiction",
 'Science': "Nonfiction",
 'Poetry': "Fiction"}

books["simple_categories"] = books["categories"].map(category_mapping)
books

# %%
#Categories with known label
books[~(books["simple_categories"].isna())] #3743

# %%
#Zero Shot Classification
fiction_categories = ["Fiction", "Nonfiction"]
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# %%
#Testing -> Will give probability for fiction and non fiction
sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[0]
pipe(sequence, fiction_categories)

# %%
#Choosing Max probability
max_index = np.argmax(pipe(sequence, fiction_categories)["scores"])
max_label = pipe(sequence, fiction_categories)["labels"][max_index]
max_label

# %%
#Function to get probability and return max one 
def generate_predictions(sequence, categories):
    predictions = pipe(sequence, categories)
    max_index = np.argmax(predictions["scores"])
    max_label = predictions["labels"][max_index]
    return max_label

# %%
#Comparing known label with sizable input of fiction and non-fiction to know how good it is
actual_cats = []
predicted_cats = []
#Fiction
for i in tqdm(range(0, 300)):
    sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Fiction"]

# %%
#Non Fiction
for i in tqdm(range(0, 300)):
    sequence = books.loc[books["simple_categories"] == "Nonfiction", "description"].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Nonfiction"]

# %%
#Checking
predictions_df = pd.DataFrame({"actual_categories": actual_cats, "predicted_categories": predicted_cats})
predictions_df
predictions_df["correct_prediction"] = (
    np.where(predictions_df["actual_categories"] == predictions_df["predicted_categories"], 1, 0)
)
predictions_df["correct_prediction"].sum() / len(predictions_df) #78% correct

# %%
#Main working on subset where category is missing
isbns = []
predicted_cats = []

missing_cats = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)
for i in tqdm(range(0, len(missing_cats))):
    sequence = missing_cats["description"][i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    isbns += [missing_cats["isbn13"][i]]

# %%
#Checking the subset after running
missing_predicted_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_cats})
missing_predicted_df

# %%
#Merging with the original
books = pd.merge(books, missing_predicted_df, on="isbn13", how="left")
books["simple_categories"] = np.where(books["simple_categories"].isna(), books["predicted_categories"], books["simple_categories"])
books = books.drop(columns = ["predicted_categories"])
books
#Not going detail inside fiction because we don't have data to cross check

# %%
#Saving updated one
books.to_csv("data/processed/books_with_categories.csv", index=False)
# %%
