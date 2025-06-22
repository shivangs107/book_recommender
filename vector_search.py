# %%
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# %%
from dotenv import load_dotenv
# Load environment variables
load_dotenv()  # Automatically finds .env in your current directory

# %%
import pandas as pd
books = pd.read_csv("books_cleaned.csv")
books
# %%
books["tagged_description"]
# %%
books["tagged_description"].to_csv("tagged_description.txt",
                                   sep = "\n",
                                   index = False,
                                   header = False)
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
# %%
documents[0]
# %%
db_books = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings())
query = "A book to teach children about nature"
docs = db_books.similarity_search(query, k = 10)
docs
# %%
books[books["isbn13"] == int(docs[0].page_content.split()[0].strip())]
# %%
def retrieve_semantic_recommendations(
        query: str,
        top_k: int = 10,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k = 50)

    books_list = []

    for i in range(0, len(recs)):
        books_list += [int(recs[i].page_content.strip('"').split()[0])]

    return books[books["isbn13"].isin(books_list)]
retrieve_semantic_recommendations("A book to teach children about nature")
# %%
