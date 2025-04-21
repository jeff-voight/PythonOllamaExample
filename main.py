import json
import os.path

from numpy.linalg import norm
from ollama import Client
import time
import numpy as np

def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append(" ".join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append(" ".join(buffer))
        return paragraphs

def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings

    embeddings = [
        client.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
    based on snippets of text provided in context. Answer only using the context provided,
    being as concise as possible. If you're unsure, just say that you don't know.
    Context:
    """

    filename = "peterpan.txt"
    paragraphs = parse_file(filename)
    start = time.perf_counter()
    embeddings = get_embeddings(filename,'nomic-embed-text', paragraphs)
    print(time.perf_counter() - start)
    print(len(embeddings))
    prompt = "who is the story's primary villain?"
    prompt2 = "how did peter defeat the villain?"

    prompt_embedding = client.embeddings(model='nomic-embed-text', prompt=prompt)[
        "embedding"
    ]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:25]
    for item in most_similar_chunks:
        print(item[0], paragraphs[item[1]])
    print("\n\n\n")
    response = client.chat(
        'gemma3:1b',
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                           + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream = False
    )
    print(response)

    print("\n\n* * * * * * * *\n\n")
    prompt2 = "how did peter defeat the villain?"

    prompt_embedding = client.embeddings(model='nomic-embed-text', prompt=prompt2)[
        "embedding"
    ]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    for item in most_similar_chunks:
        print(item[0], paragraphs[item[1]])

    print("\n\n\n")

    response2 = client.chat(
        'gemma3:1b',
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                           + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
            },
            {
                "role": "user",
                "content": prompt2
            }
        ],
        stream = False
    )
    print(response2)


client = Client(
    host='http://192.168.137.186:11434',
)
if __name__ == "__main__":
    main()

