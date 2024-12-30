!pip install openai pinecone-client python-dotenv

from dotenv import load_dotenv
import os
import time
import openai
from pinecone import Pinecone, ServerlessSpec
from openai.error import RateLimitError, OpenAIError

# Step 1: Load environment variables securely
load_dotenv()


# Step 1: Set your API keys
openai_api_key = os.getenv('OPENAI_API_KEY', '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')  # Replace with your OpenAI API key
pinecone_api_key = os.getenv('PINECONE_API_KEY', 'pcsk_76bHSj_Lr5jSmqiStQbqT55FpFhQdmcWwDR1TXdjD3GXjbpUrAR9zGdFdJJSKuigq4yJGL')  # Replace with your Pinecone API key

# Initialize Pinecone using Pinecone class
pinecone = Pinecone(api_key=pinecone_api_key, environment="us-east-1")  # Initializing Pinecone client

# Step 2: Create or connect to Pinecone index
index_name = 'business-qa-index'
if index_name not in pinecone.list_indexes().names(): # Accessing names attribute from list_indexes result
    pinecone.create_index(index_name, dimension=1536, metric='cosine')

index = pinecone.Index(index_name)

# Function to generate embeddings
def get_openai_embeddings_v1(texts, batch_size=1, retry_delay=2, max_retries=3):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        retries = 0
        while retries < max_retries:
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                embeddings.extend([data['embedding'] for data in response['data']])
                break
            except RateLimitError:
                retries += 1
                print(f"Rate limit hit. Retrying in {retry_delay} seconds ({retries}/{max_retries})...")
                time.sleep(retry_delay)
            except OpenAIError as e:
                print(f"OpenAI error: {e}")
                return []
    return embeddings

# Step 3: Insert documents into Pinecone
documents = [
    "Our company provides innovative cloud-based solutions for enterprise data management.",
    "We offer a wide range of consulting services to help businesses scale and optimize their processes.",
    "Our software helps automate workflows and improve collaboration within teams."
]

document_embeddings = get_openai_embeddings_v1(documents)
if document_embeddings:
    vectors_to_insert = [
        (str(i), doc_emb, {"text": doc}) for i, (doc_emb, doc) in enumerate(zip(document_embeddings, documents))
    ]
    index.upsert(vectors=vectors_to_insert)
    print("Documents successfully inserted into Pinecone.")
else:
    print("Error: Unable to generate embeddings for documents.")

# Step 4: Retrieve relevant documents
def retrieve_relevant_docs(query, top_k=3):
    query_embedding = get_openai_embeddings_v1([query])
    if not query_embedding:
        print("Error: Unable to generate embeddings for query.")
        return []
    results = index.query(query_embedding[0], top_k=top_k, include_metadata=True)
    return results.get('matches', [])

# Step 5: Generate an answer using OpenAI
def generate_answer(query):
    relevant_docs = retrieve_relevant_docs(query)
    if not relevant_docs:
        return "No relevant documents found."

    context = "\n".join([match['metadata']['text'] for match in relevant_docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    retries = 0
    while retries < 3:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
        except RateLimitError:
            retries += 1
            print(f"Rate limit hit. Retrying in 5 seconds ({retries}/3)...")
            time.sleep(5)
        except OpenAIError as e:
            print(f"OpenAI error: {e}")
            return "Error generating answer."
    return "Failed to generate answer after retries."

# Step 6: Test the functionality
query = "What services does the company offer?"
answer = generate_answer(query)
print("Answer:", answer)
