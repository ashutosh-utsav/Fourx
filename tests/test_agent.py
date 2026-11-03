"""Tests with a scripted fake model: no API keys, no network.

Each test gets its own empty agent.db and ledger.db in a temp folder.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

import ledger
from src import tools
from src.loop import Agent, recover, reference_for
from src.providers import gemini as gemini_provider
from src.providers import groq as groq_provider
from src.providers.base import new_call_id
from src.store import Store

# ------------------------------------------------------------------ helpers


def call(name, **args):
    return {"role": "assistant", "text": "", "tool_calls": [{"id": new_call_id(), "name": name, "args": args, "meta": {}}]}


def calls(*pairs):
    return {"role": "assistant", "text": "",
            "tool_calls": [{"id": new_call_id(), "name": n, "args": a, "meta": {}} for n, a in pairs]}


def say(text):
    return {"role": "assistant", "text": text, "tool_calls": []}


class FakeProvider:
    """Returns the scripted replies in order, and records the history it was given."""

    def __init__(self, name="gemini", script=(), delay=0.0):
        self.name, self.model, self.script, self.delay, self.seen = name, "fake", list(script), delay, []

    def complete(self, system, history, tools_):
        self.seen.append(json.loads(json.dumps(history)))
        reply = self.script.pop(0)
        if self.delay:
            time.sleep(self.delay)
        return reply


def assert_history_valid(history):
    """Every tool call has exactly one result, right after it (what both providers require)."""
    open_ids = set()
    for m in history:
        if m["role"] == "tool":
            assert m["tool_call_id"] in open_ids, f"result without a call: {m}"
            open_ids.remove(m["tool_call_id"])
        else:
            assert not open_ids, f"tool calls without results before {m}: {open_ids}"
            open_ids = {c["id"] for c in m.get("tool_calls") or []}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "DB_PATH", tmp_path / "ledger.db")  # never touch the real ledger
    store = Store(tmp_path / "agent.db")
    store.open_conversation("t", "gemini")
    return store


def make_agent(store, *providers):
    by_name = {p.name: p for p in providers}
    lines = []
    agent = Agent(store, "t", out=lines.append, provider_factory=lambda n: by_name[n])
    return agent, lines


def posted(invoice_number):
    return ledger.entries(invoice_number=invoice_number)


# ------------------------------------------------------------------ tools and guards


def test_read_tools():
    assert len(tools.list_invoices()["invoices"]) == 6
    assert tools.read_invoice("invoice-06")["total"] == "21948.00"
    assert tools.lookup_vendor("bluefin")["matches"][0]["vendor_id"] == "V-002"
    assert tools.lookup_vendor("V-004")["matches"][0]["status"] == "blocked"


@pytest.mark.parametrize("bad", ["../.env", "invoice-1", "invoice-01.json", "invoice-99", "", None, 7])
def test_bad_invoice_ids_are_rejected(bad):
    assert tools.run_tool("read_invoice", {"invoice_id": bad})["status"] == "error"
    assert tools.prepare_post(bad)["ok"] is False


def test_bad_tool_calls_become_errors_not_crashes():
    assert tools.run_tool("delete_ledger", {})["status"] == "error"
    assert tools.run_tool("read_invoice", {})["status"] == "error"
    assert tools.run_tool("read_invoice", {"invoice_id": "invoice-01", "x": 1})["status"] == "error"


def test_prepare_post_builds_details_from_the_files(env):
    check = tools.prepare_post("invoice-01")
    assert check == {
        "ok": True,
        "vendor_name": "Shree Ganesh Packaging Pvt Ltd",
        "details": {"vendor_id": "V-001", "invoice_number": "SGP/26-27/0412", "invoice_date": "2026-10-03",
                    "currency": "INR", "amount": "16862.20"},
    }


def test_blocked_vendor_is_refused(env):
    check = tools.prepare_post("invoice-04")
    assert not check["ok"] and "blocked" in check["reason"] and "Bank details" in check["reason"]


def test_duplicate_is_refused(env):
    check = tools.prepare_post("invoice-05")
    assert not check["ok"] and "JE-000001" in check["reason"]


def test_bad_arithmetic_and_unknown_vendor_are_refused(env, tmp_path, monkeypatch):
    inv = json.loads((tools.INVOICES_DIR / "invoice-01.json").read_text())
    folder = tmp_path / "invoices"
    folder.mkdir()
    (folder / "invoice-01.json").write_text(json.dumps({**inv, "total": "99999.00"}))
    (folder / "invoice-02.json").write_text(json.dumps({**inv, "vendor_name": "Shree Ganesh"}))
    (folder / "invoice-03.json").write_text(json.dumps({**inv, "lines": [], "subtotal": "0", "taxes": [], "total": "0"}))
    monkeypatch.setattr(tools, "INVOICES_DIR", folder)
    assert "does not add up" in tools.prepare_post("invoice-01")["reason"]
    assert "not in the vendor list" in tools.prepare_post("invoice-02")["reason"]
    assert not tools.prepare_post("invoice-03")["ok"]


# ------------------------------------------------------------------ the 7 assignment cases, with a fake model


def test_1_approve_posts_exactly_once(env):
    fake = FakeProvider(script=[call("read_invoice", invoice_id="invoice-01"),
                                call("post_invoice", invoice_id="invoice-01"), say("Posted.")])
    agent, lines = make_agent(env, fake)
    agent.send("post invoice-01")
    approval = env.open_approval("t")
    assert approval["details"]["amount"] == "16862.20"
    assert "V-001" in lines[-1] and "SGP/26-27/0412" in lines[-1] and "2026-10-03" in lines[-1]
    assert posted("SGP/26-27/0412") == []  # nothing posted before approval

    agent.approve()
    assert len(posted("SGP/26-27/0412")) == 1
    agent.approve()  # a second /approve does nothing
    assert len(posted("SGP/26-27/0412")) == 1
    result = fake.seen[-1][-1]["result"]
    assert result["status"] == "posted" and result["entry_id"].startswith("JE-")
    assert_history_valid(env.messages("t"))


def test_2_approval_survives_a_restart(env, tmp_path):
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-02")])
    agent, _ = make_agent(env, fake)
    agent.send("post invoice-02")
    env.conn.close()  # "quit"

    store = Store(tmp_path / "agent.db")  # "start again"
    recover(store)
    fake2 = FakeProvider(script=[say("Posted as ...")])
    agent2, lines = make_agent(store, fake2)
    agent2.resume()
    assert "BL-88213" in lines[-1] and "70800.00" in lines[-1]
    agent2.approve()
    assert len(posted("BL-88213")) == 1
    assert_history_valid(store.messages("t"))


def test_3_decline_tells_the_model_why(env):
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-03"), say("Not posted.")])
    agent, _ = make_agent(env, fake)
    agent.send("post invoice-03")
    agent.decline("PO not received yet")
    assert posted("KOS-2026-1187") == []
    result = fake.seen[-1][-1]["result"]
    assert result["status"] == "declined" and result["reason"] == "PO not received yet"
    assert env.open_approval("t") is None


def test_4_switching_providers_keeps_the_history(env):
    g = FakeProvider("gemini", [call("read_invoice", invoice_id="invoice-06"), say("Stretch wrap."),
                                say("One line.")])
    q = FakeProvider("groq", [say("Shree Ganesh, INR 21948.00")])
    agent, _ = make_agent(env, g, q)
    agent.send("what's on invoice-06?")
    agent.switch("groq")
    agent.send("which vendor and total?")
    assert q.seen[0][1]["tool_calls"][0]["name"] == "read_invoice"  # groq saw gemini's tool call
    agent.switch("gemini")
    agent.send("how many lines?")
    assert len(g.seen[-1]) == 7  # everything, including groq's answer
    assert_history_valid(env.messages("t"))


def test_5_stop_while_waiting_on_the_model(env):
    slow = FakeProvider(script=[say("never seen"), say("I read none.")], delay=5)
    agent, lines = make_agent(env, slow)
    threading.Timer(0.3, agent.stop_event.set).start()
    started = time.monotonic()
    agent.send("read every invoice")
    assert time.monotonic() - started < 1.5
    assert lines[-1] == "Stopped."
    slow.delay = 0
    agent.send("which invoices did you get through?")
    assert lines[-1] == "agent> I read none."
    assert_history_valid(env.messages("t"))
    assert len(ledger.entries()) == 1


def test_5_stop_while_running_tools(env, monkeypatch):
    monkeypatch.setenv("AGENT_TOOL_DELAY", "2")
    fake = FakeProvider(script=[calls(*[("read_invoice", {"invoice_id": f"invoice-0{i}"}) for i in range(1, 7)]),
                                say("I got through one.")])
    agent, lines = make_agent(env, fake)
    threading.Timer(2.5, agent.stop_event.set).start()
    started = time.monotonic()
    agent.send("read every invoice and summarise each one")
    assert time.monotonic() - started < 3.5
    results = [m["result"] for m in env.messages("t") if m["role"] == "tool"]
    assert results[0]["invoice_id"] == "invoice-01"
    assert all(r["status"] == "cancelled" for r in results[2:])
    monkeypatch.setenv("AGENT_TOOL_DELAY", "0")
    agent.send("which did you get through?")
    assert lines[-1] == "agent> I got through one."
    assert_history_valid(env.messages("t"))


def test_6_blocked_vendor_no_approval(env):
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-04"), say("Blocked.")])
    agent, _ = make_agent(env, fake)
    agent.send("post invoice-04")
    assert env.open_approval("t") is None
    assert fake.seen[-1][-1]["result"]["status"] == "refused"
    assert posted("OFS/OCT/26/031") == []


def test_7_duplicate_no_approval(env):
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-05"), say("Already posted.")])
    agent, _ = make_agent(env, fake)
    agent.send("post invoice-05")
    assert env.open_approval("t") is None
    assert "JE-000001" in fake.seen[-1][-1]["result"]["reason"]
    assert len(posted("BL-88190")) == 1


# ------------------------------------------------------------------ awkward moments


def test_only_one_post_waits_at_a_time(env):
    fake = FakeProvider(script=[calls(("post_invoice", {"invoice_id": "invoice-01"}),
                                      ("post_invoice", {"invoice_id": "invoice-01"}),
                                      ("post_invoice", {"invoice_id": "invoice-02"})), say("ok")])
    agent, _ = make_agent(env, fake)
    agent.send("post invoice-01 and invoice-02")
    agent.approve()
    assert len(posted("SGP/26-27/0412")) == 1 and posted("BL-88213") == []
    statuses = [m["result"]["status"] for m in env.messages("t") if m["role"] == "tool"]
    assert sorted(statuses) == ["not_run", "not_run", "posted"]
    assert_history_valid(env.messages("t"))


def test_chat_is_blocked_while_an_approval_waits(env):
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-01")])
    agent, lines = make_agent(env, fake)
    agent.send("post invoice-01")
    agent.send("hello?")
    assert "approval is waiting" in lines[-1]


def test_recheck_on_approve_catches_a_post_made_meanwhile(env):
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-01"), say("Already there.")])
    agent, _ = make_agent(env, fake)
    agent.send("post invoice-01")
    ledger.post(vendor_id="V-001", invoice_number="SGP/26-27/0412", invoice_date="2026-10-03",
                currency="INR", amount="16862.20")  # someone else posts it while we wait
    agent.approve()
    assert len(posted("SGP/26-27/0412")) == 1
    assert fake.seen[-1][-1]["result"]["status"] == "refused"


def test_invoice_changed_after_approval_is_not_posted(env):
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-01"), say("Changed.")])
    agent, _ = make_agent(env, fake)
    agent.send("post invoice-01")
    approval = env.open_approval("t")
    env.conn.execute("UPDATE approvals SET details = ? WHERE id = ?",
                     (json.dumps({**approval["details"], "amount": "1.00"}), approval["id"]))
    env.conn.commit()
    agent.approve()
    assert posted("SGP/26-27/0412") == []
    assert "changed" in fake.seen[-1][-1]["result"]["reason"]


def test_crash_after_ledger_saved_but_before_we_recorded_it(env):
    """The program dies inside ledger.post, after the entry was saved: recovery finds it."""
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-01")])
    agent, _ = make_agent(env, fake)
    agent.send("post invoice-01")
    approval = env.open_approval("t")
    env.set_approval_status(approval["id"], "posting")
    ledger.post(**approval["details"], reference=reference_for(approval["id"]))  # then "crash"

    lines = []
    recover(env, lines.append)
    assert "had been posted" in lines[0]
    assert env.open_approval("t") is None
    assert len(posted("SGP/26-27/0412")) == 1
    assert env.messages("t")[-1]["result"]["status"] == "posted"
    assert_history_valid(env.messages("t"))


def test_crash_before_the_ledger_saved_anything(env):
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-01")])
    agent, _ = make_agent(env, fake)
    agent.send("post invoice-01")
    approval = env.open_approval("t")
    env.set_approval_status(approval["id"], "posting")  # then "crash" before ledger.post saved
    lines = []
    recover(env, lines.append)
    assert "waiting for approval again" in lines[0]
    assert env.open_approval("t")["status"] == "waiting"
    assert posted("SGP/26-27/0412") == []


def test_crash_in_the_middle_of_a_tool_call(env):
    env.add_message("t", {"role": "user", "text": "read invoice-01"})
    env.add_message("t", call("read_invoice", invoice_id="invoice-01"))  # then "crash"
    fake = FakeProvider(script=[say("ok")])
    agent, _ = make_agent(env, fake)
    agent.resume()
    agent.send("hello")
    assert_history_valid(fake.seen[0])


def test_ctrl_c_during_a_post_lets_it_finish(env):
    fake = FakeProvider(script=[call("post_invoice", invoice_id="invoice-01"), say("never")])
    agent, lines = make_agent(env, fake)
    agent.send("post invoice-01")
    threading.Timer(0.5, agent.stop_event.set).start()
    agent.approve()
    assert len(posted("SGP/26-27/0412")) == 1
    assert env.approvals_with_status("posted")
    assert lines[-1] == "Stopped."
    assert_history_valid(env.messages("t"))


def test_provider_error_keeps_the_conversation_usable(env):
    class Broken(FakeProvider):
        def complete(self, *a):
            raise RuntimeError("503 overloaded")

    agent, lines = make_agent(env, Broken())
    agent.send("hi")
    assert "503" in lines[-1]
    assert_history_valid(env.messages("t"))


class RateLimited(Exception):
    pass


class FlakyProvider(FakeProvider):
    """Fails with a 'rate limit' `failures` times, then follows the script."""

    def __init__(self, failures, wait, **kw):
        super().__init__(**kw)
        self.failures, self.wait = failures, wait

    def complete(self, *a):
        if self.failures:
            self.failures -= 1
            raise RateLimited("429")
        return super().complete(*a)

    def retry_delay(self, e, attempt):
        return self.wait if isinstance(e, RateLimited) else None


def test_rate_limit_is_retried(env):
    flaky = FlakyProvider(failures=2, wait=0.1, script=[say("done")])
    agent, lines = make_agent(env, flaky)
    agent.send("hi")
    assert lines[-1] == "agent> done"
    assert sum("retrying" in line for line in lines) == 2


def test_stop_during_a_rate_limit_wait(env):
    flaky = FlakyProvider(failures=1, wait=30, script=[say("done")])
    agent, lines = make_agent(env, flaky)
    threading.Timer(0.3, agent.stop_event.set).start()
    started = time.monotonic()
    agent.send("hi")
    assert time.monotonic() - started < 1.5
    assert lines[-1] == "Stopped."


def test_gemini_retry_delay_follows_the_server():
    from google.genai import errors

    e = errors.ClientError(429, {"error": {"code": 429, "message": "Quota exceeded.", "status": "RESOURCE_EXHAUSTED",
                                           "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                        "retryDelay": "18s"}]}})
    assert gemini_provider.GeminiProvider.retry_delay(e, 0) == 19.0
    assert gemini_provider.GeminiProvider.retry_delay(errors.ClientError(400, {"error": {}}), 0) is None
    daily = errors.ClientError(429, {"error": {"code": 429, "message": "Quota", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
    assert gemini_provider.GeminiProvider.retry_delay(daily, 0) is None  # no point waiting


# ------------------------------------------------------------------ provider translation


MIXED_HISTORY = [
    {"role": "user", "text": "what's on invoice-06?"},
    {"role": "assistant", "text": "", "provider": "gemini",
     "tool_calls": [{"id": "call_a", "name": "read_invoice", "args": {"invoice_id": "invoice-06"}, "meta": {}},
                    {"id": "call_b", "name": "lookup_vendor", "args": {"query": "Shree"}, "meta": {}}]},
    {"role": "tool", "tool_call_id": "call_a", "name": "read_invoice", "result": {"total": "21948.00"}},
    {"role": "tool", "tool_call_id": "call_b", "name": "lookup_vendor", "result": {"matches": []}},
    {"role": "assistant", "text": "Stretch wrap, INR 21948.00.", "tool_calls": [], "provider": "gemini"},
    {"role": "user", "text": "which vendor?"},
    {"role": "assistant", "text": "", "provider": "groq",
     "tool_calls": [{"id": "fc_x", "name": "check_ledger", "args": {"invoice_id": "invoice-06"}, "meta": {}}]},
    {"role": "tool", "tool_call_id": "fc_x", "name": "check_ledger", "result": {"posted": False}},
    {"role": "assistant", "text": "Shree Ganesh.", "tool_calls": [], "provider": "groq"},
    {"role": "user", "text": "how many lines?"},
]


def test_to_gemini():
    contents = gemini_provider.to_gemini(MIXED_HISTORY)
    roles = [c.role for c in contents]
    assert roles == ["user", "model", "user", "model", "user", "model", "user", "model", "user"]
    assert [p.function_response.name for p in contents[2].parts] == ["read_invoice", "lookup_vendor"]
    assert contents[5].parts[0].function_call.name == "check_ledger"  # groq's call, in gemini format
    assert contents[8].parts[-1].text == "how many lines?"


def test_to_groq():
    messages = groq_provider.to_groq("sys", MIXED_HISTORY)
    assert messages[0] == {"role": "system", "content": "sys"}
    assert [t["id"] for t in messages[2]["tool_calls"]] == ["call_a", "call_b"]  # gemini's calls, groq format
    assert messages[3] == {"role": "tool", "tool_call_id": "call_a", "content": '{"total": "21948.00"}'}
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {"invoice_id": "invoice-06"}


def test_from_gemini_keeps_thought_signature():
    from google.genai import types

    response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[
        types.Part(text="Let me look."),
        types.Part(function_call=types.FunctionCall(name="read_invoice", args={"invoice_id": "invoice-01"}),
                   thought_signature=b"sig"),
    ]))])
    msg = gemini_provider.from_gemini(response)
    assert msg["text"] == "Let me look."
    assert msg["tool_calls"][0]["args"] == {"invoice_id": "invoice-01"}
    contents = gemini_provider.to_gemini([{"role": "user", "text": "x"}, msg])
    assert contents[1].parts[1].thought_signature == b"sig"  # sent back unchanged


def test_from_gemini_empty_answer():
    from google.genai import types

    msg = gemini_provider.from_gemini(types.GenerateContentResponse(candidates=[]))
    assert msg["tool_calls"] == [] and "no answer" in msg["text"]


def test_from_groq():
    from types import SimpleNamespace as NS

    tc = NS(id="fc_1", function=NS(name="read_invoice", arguments='{"invoice_id": "invoice-02"}'))
    bad = NS(id="fc_2", function=NS(name="read_invoice", arguments="{not json"))
    response = NS(choices=[NS(message=NS(content=None, tool_calls=[tc, bad]))])
    msg = groq_provider.from_groq(response)
    assert msg["tool_calls"][0] == {"id": "fc_1", "name": "read_invoice", "args": {"invoice_id": "invoice-02"}, "meta": {}}
    assert msg["tool_calls"][1]["args"] == {}
