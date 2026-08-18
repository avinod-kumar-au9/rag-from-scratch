import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import chromadb


load_dotenv()


# Create and return the OpenAI-compatible client for the Cerebras API.
def create_openai_client() -> OpenAI:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is not set.")

    return OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
    )


# Read the source document from disk so it can be processed.
def load_text_from_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as handle:
        return handle.read()


# Split the document into smaller chunks that are easier to index and retrieve.
def split_text_into_chunks(text: str, source_name: str):
    paragraphs = [paragraph.strip() for paragraph in text.strip().split("\n")]

    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) < 50:
            continue
        if paragraph.startswith("===="):
            continue
        chunks.append({
            "text": paragraph,
            "source": source_name,
        })

    return chunks


# Create or reset the Chroma collection used for storing document embeddings.
def initialize_chroma_collection(chroma_client, collection_name: str):
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    return chroma_client.create_collection(name=collection_name)


# Store the chunked text in the vector database for semantic retrieval.
def index_text_chunks(vector_collection, chunks):
    documents = [chunk["text"] for chunk in chunks]
    ids = [f"chunk_{index}" for index in range(len(chunks))]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]

    vector_collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )


# Build a reusable RAG agent that can query company documents through a tool call.
def build_rag_agent(openai_client, vector_collection):
    # Retrieve relevant chunks from the vector store for a given query.
    def search_company_documents(query: str, n_results: int = 3) -> str:
        results = vector_collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        chunks = results["documents"][0]
        return "\n\n".join(chunks)

    document_search_tool = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search company documents for HR, product, security, onboarding, and engineering information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant document content",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    tool_handlers = {
        "search_docs": search_company_documents,
    }

    # Run the agent loop: ask the model, use tools if needed, and return the answer.
    def run_rag_agent(question: str, verbose: bool = True) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful company assistant. "
                    "Use the search_docs tool to answer questions from internal documents. "
                    "If the documents do not contain the answer, say so clearly. "
                    "Base your answer only on the retrieved documents."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        if verbose:
            print(f"\n{'═' * 60}")
            print(f"🧑 Question: {question}")
            print(f"{'─' * 60}")

        for _ in range(3):
            response = openai_client.chat.completions.create(
                model="gpt-oss-120b",
                messages=messages,
                tools=document_search_tool,
                temperature=0.2,
            )

            choice = response.choices[0]

            if choice.finish_reason == "stop":
                if verbose:
                    print(f"🤖 Answer: {choice.message.content}")
                    print(f"{'═' * 60}")
                return choice.message.content

            if choice.message.tool_calls:
                messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if verbose:
                        print(f"  🔍 Searching: \"{args['query']}\"")

                    result = tool_handlers[func_name](**args)

                    if verbose:
                        print(f"  📄 Retrieved {len(result)} chars of context")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

        return "Could not answer within step limit."

    return run_rag_agent


if __name__ == "__main__":
    # Main workflow: connect to the API, load the document, index it, and ask a question.
    openai_client = create_openai_client()
    chroma_client = chromadb.Client()
    vector_collection = initialize_chroma_collection(chroma_client, "company_docs")

    document_text = load_text_from_file("hr_document.txt")
    chunks = split_text_into_chunks(document_text, "HR Policy")
    index_text_chunks(vector_collection, chunks)

    rag_agent = build_rag_agent(openai_client, vector_collection)
    rag_agent("How many sick leave days do I get per year?")
