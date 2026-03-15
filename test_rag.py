from query_data import query_rag
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
import os

# Configuration - Change this to toggle between modes
USE_ONLINE_MODE = False  # Set to True for ChatGPT API, False for Ollama/Mistral
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Store your API key in environment variable

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true', 'false', or 'unknown') Does the actual response match the expected response?
If the actual response indicates the model couldn't find information (e.g., "I don't know", "not found", "no information"), answer 'unknown'.
"""


def test_imf_mission_statement():
    assert query_and_validate(
        question="What is the primary mission of the IMF?",
        expected_response="promote international monetary cooperation and financial stability",
    )


def test_peter_favorite_color():
    assert query_and_validate(
        question="What is Peter's favorite color?",
        expected_response="Peter's favorite color is green",
    )


def test_imf_voting_rights():
    assert query_and_validate(
        question="What determines voting rights in the IMF?",
        expected_response="quota contributions",
    )


def test_imf_special_drawing_rights():
    assert query_and_validate(
        question="What are Special Drawing Rights (SDRs)?",
        expected_response="an international reserve asset created by the IMF",
    )


def test_imf_surveillance_function():
    assert query_and_validate(
        question="What is Article IV surveillance?",
        expected_response="IMF monitoring of member countries' economic policies",
    )


def test_imf_lending_facilities():
    assert query_and_validate(
        question="What is the Stand-By Arrangement used for?",
        expected_response="short-term balance of payments financing",
    )


def test_peter_favorite_candy():
    assert query_and_validate(
        question="What is Peter's favorite candy?",
        expected_response="Peter's favorite candy is swedish fish",
    )


def get_llm_model():
    """Returns the appropriate LLM model based on configuration"""
    if USE_ONLINE_MODE:
        return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    else:
        return Ollama(model="mistral")


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = get_llm_model()
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    elif "unknown" in evaluation_results_str_cleaned:
        # Print response in Yellow if information was not found.
        print("\033[93m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true', 'false', or 'unknown'."
        )

