"""The tools the model can ask for, and the checks that guard posting.

Plain Python, no model code. The model only ever gives an invoice id: everything
that gets posted (vendor id, number, date, amount) is read from the files here.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import ledger

ROOT = Path(__file__).resolve().parent.parent
INVOICES_DIR = ROOT / "invoices"
VENDORS_PATH = ROOT / "vendors.json"
INVOICE_ID = re.compile(r"invoice-\d{2}")


class ToolError(Exception):
    """A problem to report back to the model (bad id, missing file, bad arguments)."""


class Refused(Exception):
    """Posting is not allowed. The message says why."""


# ---------------------------------------------------------------- data access


def load_vendors() -> list[dict]:
    return json.loads(VENDORS_PATH.read_text())


def load_invoice(invoice_id: str) -> dict:
    # The id must look like "invoice-01": this also stops paths like "../.env".
    if not isinstance(invoice_id, str) or not INVOICE_ID.fullmatch(invoice_id):
        raise ToolError(f"invalid invoice id {invoice_id!r}: expected something like 'invoice-01'")
    path = INVOICES_DIR / f"{invoice_id}.json"
    if not path.is_file():
        raise ToolError(f"there is no invoice {invoice_id}")
    return json.loads(path.read_text())


def vendor_by_exact_name(name: str) -> dict | None:
    """The one vendor whose name matches exactly, or None (missing or ambiguous)."""
    matches = [v for v in load_vendors() if v.get("name") == name]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------- read-only tools


def list_invoices() -> dict:
    invoices = []
    for path in sorted(INVOICES_DIR.glob("invoice-*.json")):
        inv = json.loads(path.read_text())
        invoices.append(
            {
                "invoice_id": path.stem,
                "invoice_number": inv.get("invoice_number"),
                "vendor_name": inv.get("vendor_name"),
                "invoice_date": inv.get("invoice_date"),
                "currency": inv.get("currency"),
                "total": inv.get("total"),
            }
        )
    return {"invoices": invoices}


def read_invoice(invoice_id: str) -> dict:
    # Set AGENT_TOOL_DELAY=3 to slow this down, so there is time to test stopping.
    delay = float(os.environ.get("AGENT_TOOL_DELAY", "0") or 0)
    if delay:
        time.sleep(delay)
    return {"invoice_id": invoice_id, **load_invoice(invoice_id)}


def lookup_vendor(query: str) -> dict:
    """Find vendors by id ("V-002") or by part of the name, ignoring case."""
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a vendor name or vendor id")
    q = query.strip().lower()
    matches = [v for v in load_vendors() if v["vendor_id"].lower() == q or q in v["name"].lower()]
    return {"query": query, "matches": matches}


def check_ledger(invoice_id: str) -> dict:
    """Is this invoice already in the ledger (same vendor and invoice number)?"""
    inv = load_invoice(invoice_id)
    vendor = vendor_by_exact_name(inv.get("vendor_name", ""))
    filters = {"invoice_number": inv.get("invoice_number", "")}
    if vendor:
        filters["vendor_id"] = vendor["vendor_id"]
    found = ledger.entries(**filters)
    return {
        "invoice_id": invoice_id,
        "invoice_number": filters["invoice_number"],
        "vendor_id": vendor["vendor_id"] if vendor else None,
        "posted": bool(found),
        "entries": [
            {k: e[k] for k in ("entry_id", "vendor_id", "amount", "currency", "posted_at")} for e in found
        ],
    }


READ_TOOLS = {
    "list_invoices": list_invoices,
    "read_invoice": read_invoice,
    "lookup_vendor": lookup_vendor,
    "check_ledger": check_ledger,
}


def run_tool(name: str, args: dict) -> dict:
    """Run a read-only tool. Problems become an error result for the model, not a crash."""
    fn = READ_TOOLS.get(name)
    if fn is None:
        return {"status": "error", "error": f"unknown tool {name!r}"}
    try:
        return fn(**(args or {}))
    except ToolError as e:
        return {"status": "error", "error": str(e)}
    except TypeError as e:  # wrong or missing arguments from the model
        return {"status": "error", "error": f"bad arguments for {name}: {e}"}


# ---------------------------------------------------------------- posting guards


def _money(value, label: str) -> Decimal:
    try:
        d = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"{label} is not a number: {value!r}") from None
    if not d.is_finite():
        raise ValueError(f"{label} is not a number: {value!r}")
    return d


def arithmetic_problems(inv: dict) -> list[str]:
    """Check the invoice adds up, with exact decimals. Returns the problems found."""
    problems = []
    lines = inv.get("lines") or []
    if not lines:
        problems.append("the invoice has no lines")
    line_sum = Decimal(0)
    for i, line in enumerate(lines, 1):
        qty = _money(line["quantity"], f"line {i} quantity")
        price = _money(line["unit_price"], f"line {i} unit price")
        amount = _money(line["amount"], f"line {i} amount")
        if qty * price != amount:
            problems.append(f"line {i}: {qty} x {price} is not {amount}")
        line_sum += amount
    subtotal = _money(inv["subtotal"], "subtotal")
    if line_sum != subtotal:
        problems.append(f"lines add up to {line_sum}, not the subtotal {subtotal}")
    tax = sum((_money(t["amount"], "tax amount") for t in inv.get("taxes") or []), Decimal(0))
    total = _money(inv["total"], "total")
    if subtotal + tax != total:
        problems.append(f"subtotal {subtotal} + taxes {tax} is not the total {total}")
    if total <= 0:
        problems.append(f"the total {total} is not positive")
    return problems


def prepare_post(invoice_id) -> dict:
    """Decide whether an invoice may be posted, and build exactly what would be posted.

    Returns {"ok": True, "details": {...}, "vendor_name": ...} or {"ok": False, "reason": ...}.
    Anything unclear is refused: we never guess when money is involved.
    """

    def refuse(reason: str) -> dict:
        return {"ok": False, "reason": reason}

    try:
        inv = load_invoice(invoice_id)
    except ToolError as e:
        return refuse(str(e))
    try:
        number = inv["invoice_number"]
        vendor_name = inv["vendor_name"]
        invoice_date = inv["invoice_date"]
        currency = inv["currency"]
        date.fromisoformat(invoice_date)
        problems = arithmetic_problems(inv)
        total = _money(inv["total"], "total").quantize(Decimal("0.01"))
    except KeyError as e:
        return refuse(f"{invoice_id} is missing the field {e}")
    except ValueError as e:
        return refuse(f"{invoice_id} is malformed: {e}")
    if problems:
        return refuse(f"{invoice_id} does not add up: " + "; ".join(problems))
    if not currency:
        return refuse(f"{invoice_id} has no currency")

    vendor = vendor_by_exact_name(vendor_name)
    if vendor is None:
        return refuse(f"the vendor {vendor_name!r} is not in the vendor list (exact name match required)")
    if vendor.get("status") != "active":  # anything other than "active" is refused
        reason = vendor.get("blocked_reason") or "no reason given"
        return refuse(
            f"vendor {vendor['vendor_id']} ({vendor_name}) is {vendor.get('status')}: {reason}. "
            "Invoices from this vendor must not be posted."
        )

    existing = ledger.entries(vendor_id=vendor["vendor_id"], invoice_number=number)
    if existing:
        e = existing[0]
        return refuse(
            f"invoice {number} from {vendor['vendor_id']} is already in the ledger as "
            f"{e['entry_id']} (posted {e['posted_at']}). It must not be posted twice."
        )

    details = {
        "vendor_id": vendor["vendor_id"],
        "invoice_number": number,
        "invoice_date": invoice_date,
        "currency": currency,
        "amount": str(total),
    }
    return {"ok": True, "details": details, "vendor_name": vendor_name}


def post_checked(invoice_id: str, approved: dict, reference: str) -> str:
    """The only place that calls ledger.post. Runs every check again first.

    `approved` is exactly what the clerk saw; if the invoice now gives anything
    different, we refuse rather than post something the clerk did not approve.
    """
    check = prepare_post(invoice_id)
    if not check["ok"]:
        raise Refused(check["reason"])
    if check["details"] != approved:
        raise Refused(f"{invoice_id} has changed since it was shown for approval; ask to post it again")
    return ledger.post(**approved, reference=reference)


def entries_with_reference(reference: str) -> list[dict]:
    return ledger.entries(reference=reference)


# ---------------------------------------------------------------- tool list for the model

_INVOICE_ID_ARG = {
    "type": "object",
    "properties": {"invoice_id": {"type": "string", "description": "Invoice id, like 'invoice-01'"}},
    "required": ["invoice_id"],
}

TOOL_SPECS = [
    {
        "name": "list_invoices",
        "description": "List every invoice on file: invoice id, vendor, invoice number, date and total.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_invoice",
        "description": "Read one invoice in full: lines, taxes, subtotal and total.",
        "parameters": _INVOICE_ID_ARG,
    },
    {
        "name": "lookup_vendor",
        "description": "Look up vendors by part of their name or by vendor id (V-001). "
        "Shows the vendor id, status (active or blocked) and any blocked reason.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Vendor name (or part of it) or id"}},
            "required": ["query"],
        },
    },
    {
        "name": "check_ledger",
        "description": "Check whether an invoice is already posted in the ledger.",
        "parameters": _INVOICE_ID_ARG,
    },
    {
        "name": "post_invoice",
        "description": "Request to post one invoice to the ledger. The program checks it (blocked vendor, "
        "already posted, totals) and asks the clerk to approve; nothing is posted without that. "
        "Only call this when the clerk asked to post this invoice.",
        "parameters": _INVOICE_ID_ARG,
    },
]
