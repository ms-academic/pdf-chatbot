# text_splitter.py

from langchain.text_splitter import CharacterTextSplitter
import os

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    folder_path = "extracted_text"

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                text = file.read()

            chunks = split_text_into_chunks(text)
            print(f"\nFile: {filename}")
            print(f"Number of text chunks: {len(chunks)}")
            print("Sample chunk:\n", chunks[0] if chunks else "No content.")
