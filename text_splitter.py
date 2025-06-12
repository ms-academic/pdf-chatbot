# text_splitting.py

from langchain.text_splitter import CharacterTextSplitter

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    with open("extracted_text/sample.txt", "r") as file:
        text = file.read()

    chunks = split_text_into_chunks(text)
    print(f"Number of text chunks: {len(chunks)}")
    print("Sample chunk:\n", chunks[0])
