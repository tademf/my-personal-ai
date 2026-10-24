"""
Interactive chat with the fine-tuned model.

Run:
    python src/chat.py

Commands inside chat:
    /quit   — exit
    /clear  — reset conversation history
    /help   — show available commands

Note: this model does not support multi-turn context. Each user message
is treated as an independent prompt. History is shown for readability
but is not sent to the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate import load_model, generate_reply


# ── Banner ───────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  Personal AI Assistant — from-scratch model                  ║
║  Trained locally on your device. No internet required.       ║
╚══════════════════════════════════════════════════════════════╝

Type a message and press Enter. Commands:
    /quit    exit the chat
    /clear   clear the screen and history
    /help    show this help

"""


def print_banner() -> None:
    print(BANNER)


def print_help() -> None:
    print(
        "\nAvailable commands:\n"
        "  /quit   exit\n"
        "  /clear  clear screen and history\n"
        "  /help   show this message\n"
    )


# ── Main loop ────────────────────────────────────────────────


def main() -> None:
    print("Loading model…")
    model, tokenizer = load_model()
    print("Ready.\n")

    print_banner()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            if user_input in ("/quit", "/exit"):
                print("Bye!")
                break
            elif user_input == "/clear":
                print("\033[2J\033[H", end="")
                print_banner()
                continue
            elif user_input == "/help":
                print_help()
                continue
            else:
                print(f"Unknown command: {user_input}. Type /help for available commands.")
                continue

        # Generate reply
        try:
            reply = generate_reply(model, tokenizer, user_input)
        except Exception as e:
            print(f"Assistant: [Error: {e}]")
            continue

        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()