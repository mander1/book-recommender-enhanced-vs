import os

import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "books_with_emotions.csv")
books = pd.read_csv(csv_path)

print(books.columns)

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "CoverNotAvailable.jpg",
    books["large_thumbnail"],
)
txt_path = os.path.join(BASE_DIR, "tagged_description.txt")
raw_documents = TextLoader(txt_path, encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        intial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=intial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category and category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category]

    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprised":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return books_recs

def recommend_books(
        query: str,
        category: str,
        tone: str,
):

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_desc = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(",")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_desc}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Enter your query", placeholder = "What type of book are you looking for?")
        categories_dropdown = gr.Dropdown(label = "Select a category", choices = categories, value = "All")
        tone_dropdown = gr.Dropdown(label = "Select a tone", choices = tones, value = "All")
        submit_button = gr.Button("Submit")

    gr.Markdown("## Recommendations")
    output_gallery = gr.Gallery(label = "Recommended Books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, categories_dropdown, tone_dropdown],
                        outputs= output_gallery)

if __name__ == "__main__":
    dashboard.launch()