# AI-Powered Business Q&A System

This project demonstrates a robust Question and Answer system powered by OpenAI and Pinecone. The system uses OpenAI's `text-embedding-ada-002` model for generating document and query embeddings and Pinecone for efficient vector storage and retrieval. It is designed to retrieve relevant information from a set of business documents and generate answers to user queries.

---

## Features

- **OpenAI Integration**: Utilizes OpenAI's embedding and completion APIs for generating embeddings and answers.
- **Pinecone Integration**: Implements vector storage and retrieval using Pinecone for efficient querying of documents.
- **Rate Limiting Handling**: Includes retry mechanisms to handle API rate limits and errors gracefully.
- **Customizable**: Easily extendable for adding more documents or adjusting query behavior.
- **Secure API Key Management**: Supports environment variables for securely managing API keys.

---

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Preethamvarma/RAG-Model-for-QA-bot
    cd business-qa-system
    ```

2. **Install Dependencies**:
    ```bash
    pip install openai pinecone-client python-dotenv
    ```

3. **Set Environment Variables**:
    - Create a `.env` file in the project root:
      ```dotenv
      OPENAI_API_KEY=your_openai_api_key
      PINECONE_API_KEY=your_pinecone_api_key
      ```
    - Replace `your_openai_api_key` and `your_pinecone_api_key` with your actual API keys.

4. **Run the Script**:
    ```bash
    python script.py
    ```

---

## Usage

### Insert Documents
The script demonstrates how to insert sample business documents into the Pinecone index. You can modify the `documents` list to include your own data.

### Query Documents
Use the `generate_answer(query)` function to retrieve relevant documents and generate answers to queries.

Example:
```python
query = "What services does the company offer?"
answer = generate_answer(query)
print("Answer:", answer)

**Project Structure**
script.py: Main script containing the functionality for embedding, document insertion, querying, and answer generation.

**Error Handling**
Rate Limit Handling: Automatically retries requests in case of rate limit errors from OpenAI.
General Errors: Prints appropriate error messages for debugging.

**Customization**
Documents: Update the documents list with your business content.
Pinecone Index: Modify the index_name and dimension to match your specific use case.
Query Behavior: Adjust the top_k parameter in retrieve_relevant_docs() for the number of relevant documents retrieved.

**Dependencies**
openai
pinecone-client
python-dotenv

Contributing
Contributions are welcome! Feel free to submit issues or pull requests.
