"""Accounts payable agent for Acme Foods.

    python agent.py --conversation demo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.loop import Agent, recover
from src.providers import PROVIDERS
from src.store import Store

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "agent.db"
HELP = (
    "Commands: /model gemini|groq, /approve, /decline <reason>, /quit.\n"
    "Press Ctrl-C while the agent is working to stop it."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conversation", "-c", default="default", help="conversation name (to resume it later)")
    parser.add_argument("--model", choices=PROVIDERS, help="provider to use (default: gemini, or the saved one)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    store = Store(DB_PATH)
    recover(store)  # settle any post that was in flight when the program last died

    is_new = store.open_conversation(args.conversation, args.model or os.environ.get("AGENT_PROVIDER", "gemini"))
    agent = Agent(store, args.conversation)
    if args.model and args.model != agent.provider_name:
        agent.switch(args.model)
    state = "new" if is_new else f"resumed, {store.count_messages(args.conversation)} messages"
    print(f'Conversation "{args.conversation}" ({state}). Model: {agent.provider_name}.\n{HELP}')
    agent.resume()

    while True:
        try:
            line = input("\nyou> ").strip()
        except KeyboardInterrupt:  # nothing is running: a late Ctrl-C shouldn't quit the program
            print("\n(Nothing to stop. Type /quit to exit.)")
            continue
        except EOFError:  # Ctrl-D, or the input ran out
            print()
            break
        if not line:
            continue
        if not line.startswith("/"):
            agent.send(line)
            continue
        command, _, rest = line.partition(" ")
        rest = rest.strip()
        if command == "/quit":
            break
        elif command == "/approve":
            agent.approve()
        elif command == "/decline":
            agent.decline(rest)
        elif command == "/model":
            agent.switch(rest) if rest else print(f"Model: {agent.provider_name}")
        elif command == "/help":
            print(HELP)
        else:
            print(f"Unknown command {command}. {HELP}")
    print("Bye. Everything is saved.")


if __name__ == "__main__":
    main()
