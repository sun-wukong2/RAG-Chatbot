import argparse
import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI

from get_embedding_function import get_embedding_function

# Toggle between local and cloud Chroma
USE_LOCAL_CHROMA = True  # True = local, False = cloud

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on provided context.
You MUST include proper citations for every claim you make.

Instructions:
1. Answer the question using ONLY the provided context.
2. For every statement, cite the source document and page number.
3. Use this citation format: [Source: filename.pdf, Page X]
4. If information spans multiple pages, list all relevant pages.
5. Never make up or assume page numbers - only use what is provided in the context.

Context:
{context}

---

Question: {question}

Provide your answer with proper citations included:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model", type=str, default="mistral", choices=["mistral", "openai"],
                        help="Use Mistral or OpenAI")
    args = parser.parse_args()
    query_text = args.query_text
    response = query_rag(query_text, model=args.model)
    print(response)


def get_chroma_db():
    embedding_function = get_embedding_function()

    if USE_LOCAL_CHROMA:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function
        )
    else:
        db = Chroma(
            client_type="http",
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", 8000)),
            collection_name="documents",
            embedding_function=embedding_function,
            api_key=os.getenv("CHROMA_API_KEY")
        )

    return db



def get_model(model_type: str = "mistral"):
    if model_type == "openai":
        return ChatOpenAI(model="gpt-4", temperature=0.7)
    else:
        return Ollama(model="mistral")


def query_rag(query_text: str, model: str = "mistral"):
    db = get_chroma_db()

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([
        f"Source (File: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}): {doc.page_content}"
        for doc, _score in results
    ])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    llm = get_model(model)
    response_text = llm.invoke(prompt)

    return response_text


if __name__ == "__main__":
    main()




