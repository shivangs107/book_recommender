# Semantic Book Recommender System

![Book Recommendation Example](data/assets/cover-not-found.jpg) *(example screenshot placeholder)*

## 📖 Overview
A semantic book recommendation system that:
- Uses vector search to find relevant books based on description similarity
- Filters by categories (Fiction/Nonfiction/Children)
- Sorts by emotional tones (Happy, Sad, Angry, etc.)
- Presents results in an interactive Gradio interface

## 🏗️ Project Architecture
<pre> ```plaintext
book_recommender/
├── data/
│ ├── raw/ # Original datasets
│ ├── processed/ # Transformed data
│ └── assets/ # Static files
├── scripts/
│ └── data_processing/ # Data pipeline scripts
├── app/ # Gradio application
├── requirements.txt # Dependencies
``` </pre>

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/book_recommender.git
   cd book_recommender

## Installing Dependencies
pip install -r requirements.txt

## Run
python app/gradio_dashboard.py

## Features
Semantic Search: Find books similar to your description

Category Filtering: Fiction/Nonfiction/Children

Emotional Sorting: Happy, Sad, Surprising, etc.

Visual Gallery: 16-book display with covers

## Data Pipeline
1) Exploration & Cleaning (data_exploration.py)
        Removed books with missing descriptions
        Filtered short descriptions (<25 words)
        Output: books_cleaned.csv
2) Category Classification (text_classification.py)
        Simplified 567 categories → 5 main groups
        Zero-shot learning with bart-large-mnli (78% accuracy)
        Output: books_with_categories.csv
3) Emotion Analysis (sentiment_analysis.py)
        DistilRoBERTa emotion classification (66% accuracy)
        7 emotions: anger, joy, sadness, etc.
        Output: books_with_emotions.csv
4) Vector Search (vector_search.py)
        LangChain + HuggingFace embeddings
        Chroma vector database
        Output: tagged_description.txt

## Technical Stack
<pre> ```plaintext
Component	            Technology Used
Data Processing	                Pandas, NumPy
NLP Models	                Transformers, LangChain
Vector Database	                Chroma
Embeddings	                all-MiniLM-L6-v2
UI Framework	                Gradio
``` </pre>

## Data Insights
<pre> ```plaintext
Original: 6,810 books
After cleaning: 5,197 books
Key Features:
    ISBN13 (unique identifier)
    Title/Subtitle
    Authors
    Categories
    Description (avg. 87 words)
    Emotional scores
``` </pre>

## Recommendation Logic
def retrieve_semantic_recommendations(query, category, tone):

    # 1. Vector similarity search

    # 2. Category filtering

    # 3. Emotion-based sorting

    # Returns top 16 relevant books

