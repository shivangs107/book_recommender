# %%
#Importing Libraries
from langchain_community.document_loaders import TextLoader #Convert text into format langchain can work with
from langchain_text_splitters import CharacterTextSplitter #Spliting text into chunks as individual description of books
from langchain_community.embeddings import HuggingFaceEmbeddings #Converting chunks into embedding
from langchain_community.vectorstores import Chroma #Storing embedding (database)
import pandas as pd

# %%
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# %%
#Loading clean books data
books = pd.read_csv("data/processed/books_cleaned.csv")
books

# %%
#Created tagged_description txt file
books["tagged_description"]
books["tagged_description"].to_csv("data/processed/tagged_description.txt",
                                   sep = "\n",
                                   index = False,
                                   header = False)

# %%
# Set encoding explicitly
raw_documents = TextLoader("data/processed/tagged_description.txt", encoding="utf-8").load()

# Now split the documents
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

# %%
documents[0]

# %%
#Create document embedding and store them in vector database
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embedding)

# %%
#Testing
query = "A book to teach children about nature"
docs = db_books.similarity_search(query, k = 10)
for i, doc in enumerate(docs):
    print(f"{i+1}.", doc.page_content)

# %%
#To give isbn and other fields properly
books[books["isbn13"] == int(docs[0].page_content.split()[0].strip())]

# %%
#To get the recommendation
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
