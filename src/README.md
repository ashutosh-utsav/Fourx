# Acme Foods invoice agent

A command-line assistant for an accounts payable clerk. It reads vendor invoices, checks them, and posts them to the ledger only after the clerk approves.

**The AI suggests, the code checks, the clerk decides.**

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install google-genai groq python-dotenv pytest
python agent.py --conversation demo
```

Put your keys in `.env`:

```
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

| Command | What it does |
|---|---|
| `/approve` | Post the invoice waiting for approval |
| `/decline <reason>` | Don't post it; the agent is told why |
| `/model gemini` or `/model groq` | Switch AI, same conversation |
| `/quit` | Exit (everything is already saved) |
| Ctrl-C while it's working | Stop it |

Run the tests with `pytest -q` (no keys or internet needed).

## Providers

| Provider | Model | Change it with |
|---|---|---|
| Google Gemini | `gemini-3.5-flash-lite` | `GEMINI_MODEL` in `.env` |
| Groq | `openai/gpt-oss-120b` | `GROQ_MODEL` in `.env` |

## Files

| File | Job |
|---|---|
| `agent.py` | Start here. Reads what you type and handles commands |
| `src/loop.py` | The agent loop, approve, decline, stop, crash recovery |
| `src/tools.py` | The 5 tools and every posting check |
| `src/store.py` | Saves everything to `agent.db` |
| `src/lock.py` | Lets only one post happen at a time |
| `src/providers/` | Talks to Gemini and Groq |
| `tests/test_agent.py` | Tests with a fake AI |

## Tools the AI can use

| Tool | Does |
|---|---|
| `list_invoices` | List all invoices |
| `read_invoice` | Read one invoice |
| `lookup_vendor` | Find a vendor and its status |
| `check_ledger` | Is this invoice already posted? |
| `post_invoice` | Ask to post. Runs the checks, then asks the clerk |

The AI only gives an invoice id like `invoice-01`. The code reads the vendor, number, date and amount from the files itself, so the AI can't change what gets posted.

## Checks before posting (in code, not left to the AI)

1. The invoice id is valid and the file exists
2. The numbers add up (lines, subtotal, taxes, total)
3. The vendor is on the list (exact name match)
4. The vendor is active, not blocked
5. The invoice is not already in the ledger (same vendor and invoice number)

If any check fails, the clerk is never asked. All checks run again when the clerk approves, and the post must match exactly what the clerk saw.

## What is saved, where, when

Everything is saved to `agent.db` (SQLite) the moment it happens. It is kept separate from `ledger.db`, because `python ledger.py reset` deletes that file.

| Table | Holds |
|---|---|
| `conversations` | Each conversation and which AI it uses |
| `messages` | Every message, tool request and tool result |
| `approvals` | Each approval: `waiting` → `posting` → `posted`, or `declined` / `refused` / `failed` |

## Switching providers

Messages are saved in our own simple format, not Gemini's or Groq's. Before each call, the history is translated into the chosen provider's format. So `/model` can switch at any point, and the new AI sees the whole conversation, including earlier tool requests and results.

## Approval

When the checks pass, the approval is saved first, then shown. The agent pauses until `/approve` or `/decline`. If you quit, the approval is shown again next time. Only one approval can wait at a time.

## Stop

The AI call runs in the background while the program watches for Ctrl-C. Ctrl-C stops it within a fraction of a second. Any tool request left unanswered is marked "cancelled", so the next message works. A post that has started is never cut off halfway; it finishes first.

## Crash safety

- Before posting, the approval is marked `posting`, and the ledger entry gets a unique tag in its `reference` field.
- On start, anything still `posting` is looked up in the ledger by that tag. Found: it's recorded as posted. Not found: it goes back to waiting.
- So an invoice is never posted twice, even if the program dies in the middle of a post.

## Two terminals at once

Posting happens while holding a lock (`ledger.lock`). A second terminal waits, checks again, sees the invoice is already posted, and refuses. The operating system releases the lock if a program crashes.

## Not handled

- Posts made outside this agent can't be stopped, only caught by the check on approval.
- The Windows lock code is written but not tested.
- Free AI plans have low limits. The agent waits and retries, but a long session may need `/model` to switch.
- One clerk per conversation; no logins.

## Next steps

- Warn about near-duplicates (e.g. `BL-88190` vs `BL88190`) and unusual amounts.
- Show a history of approvals (who approved what, and when).
