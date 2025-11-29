"""
main.py

Entry point for running the Interactive Learning Companion from the command line.

Usage (from repo root):

    cd src
    python main.py

Make sure the environment variable GEMINI_API_KEY is set,
OR be ready to type your key when prompted.
"""

import os
from getpass import getpass

import google.generativeai as genai

from memory import MemorySystem
from agent import LearningAgent


def configure_gemini() -> genai.GenerativeModel:
    """
    Configure the Gemini LLM safely.

    Priority:
    1. Use GEMINI_API_KEY environment variable if available.
    2. Otherwise, ask the user to type it (hidden) via getpass().
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("Enter your Gemini API key (input will be hidden):")
        api_key = getpass()
        if not api_key:
            raise RuntimeError("No API key provided for Gemini.")

    genai.configure(api_key=api_key)

    model_name = "gemini-pro-latest"
    llm = genai.GenerativeModel(model_name)
    print(f"Gemini model '{model_name}' loaded successfully.")
    return llm


def chat_with_agent(agent: LearningAgent) -> None:
    """Simple CLI loop to interact with the learning companion."""
    print("Interactive Learning Companion – Python Tutor")
    print("Type 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower().strip() in {"quit", "exit"}:
            print("Session ended.")
            break

        # For this CLI demo we are not automatically checking correctness;
        # correctness and user_rating are left as None.
        answer = agent.handle_turn(
            user_input=user_input,
            correctness=None,
            user_rating=None,
        )
        print("\nAgent:", answer)
        print("-" * 60)


def main() -> None:
    llm = configure_gemini()
    memory = MemorySystem()
    agent = LearningAgent(llm, memory)
    chat_with_agent(agent)


if __name__ == "__main__":
    main()
