"""Post a Slack alert for the fraud analyst when an alert crosses the risk threshold.

Configured through environment variables (set them in the repo-root `.env`, which the deployment loads). Use ONE of:

    SLACK_BOT_TOKEN + SLACK_CHANNEL   a bot token (xoxb-..., scope `chat:write`) and the channel ID to post to
                                      (the bot must be invited to the channel)
    SLACK_WEBHOOK_URL                 a Slack incoming-webhook URL (posts to the channel chosen when it was created)

    AGENTSMITH_BASE_URL               deployment URL for the review snippet (defaults to LANGSMITH_HOST_API_URL,
                                      which LangSmith sets on the deployment)

If Slack is not configured the step is skipped, never failed: a Slack problem must not stop an investigation from
reaching a human. The message carries pseudonymous subject references only, no customer names. Messages go over
HTTPS only (plain http is accepted for loopback test servers).
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

MAX_LEGS_SHOWN = 12
SLACK_API_BASE = "https://slack.com/api"


def _escape(text: object) -> str:
    """Slack mrkdwn treats & < > specially; escape them so data can't inject links or mentions."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _review_section(thread_id: str | None) -> list[str]:
    """How the analyst finds and resumes the paused investigation."""
    if not thread_id:
        return []
    base_url = os.environ.get("AGENTSMITH_BASE_URL") or os.environ.get("LANGSMITH_HOST_API_URL", "")
    tenant, deployment = os.environ.get("LANGSMITH_TENANT_ID"), os.environ.get("LANGSMITH_HOST_PROJECT_ID")
    lines = ["", "*To review and decide*", f"Thread ID: `{_escape(thread_id)}`"]
    if tenant and deployment:
        url = f"https://smith.langchain.com/o/{tenant}/host/deployments/{deployment}"
        lines.append(f"<{url}|Open the deployment in LangSmith> (then Studio, and find the thread above)")
    if base_url:
        lines += [
            "Or resume it from code:",
            "```",
            "from langgraph_sdk import get_sync_client",
            f'client = get_sync_client(url="{_escape(base_url)}", api_key="<your LangSmith API key>")',
            f'client.runs.wait("{_escape(thread_id)}", "agentsmith",',
            '    command={"resume": {"decision": "investigate" | "clear", "notes": "..."}})',
            "```",
        ]
    return lines


def build_message(state: dict, thread_id: str | None = None) -> str:
    """The Slack (mrkdwn) alert for one flagged chain."""
    chain = state.get("chain_id") or "single transaction"
    legs = state.get("chain_legs", [])
    banks = sorted({l.get("bank_id", "?") for l in legs})

    lines = [
        f":rotating_light: *AgentSmith: review required, {_escape(chain)} (risk {state.get('risk_score')}/100)*",
        "",
        f"*Alert transaction:* `{_escape(state.get('alert_transaction_id'))}`",
        f"*Chain:* `{_escape(chain)}`   *Banks:* {_escape(', '.join(banks))}",
        f"*Risk score:* {state.get('risk_score')}/100 (review threshold 70)",
        "",
        "*Why it was flagged*",
        *[f"• {_escape(reason)}" for reason in state.get("risk_reasons", [])],
        "",
        "*Bank analyst assessments*",
        *[f"• {_escape(e.get('bank_id'))}: {_escape(e.get('assessment', ''))}" for e in state.get("bank_enrichments", [])],
        "",
        "*Chain legs*",
        "```",
    ]
    for leg in sorted(legs, key=lambda l: l.get("timestamp", ""))[:MAX_LEGS_SHOWN]:
        lines.append(
            f"{leg.get('timestamp', '')[11:19]}  {leg.get('bank_id')}  {leg.get('account_id')}  "
            f"{leg.get('type')}  £{leg.get('amount', 0):,.2f}  -> {leg.get('counterparty_account')}"
        )
    if len(legs) > MAX_LEGS_SHOWN:
        lines.append(f"... and {len(legs) - MAX_LEGS_SHOWN} more legs")
    lines += ["```", "", "The investigation is paused awaiting your decision: investigate or clear."]
    lines += _review_section(thread_id)
    return "\n".join(lines).replace("```\n```", "```")


def _check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")):
        raise ValueError("Slack URL must use https")


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    _check_url(url)
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, json.dumps(payload).encode(), headers, method="POST")
    with urllib.request.urlopen(request, timeout=15) as response:  # verifies TLS certificates by default
        body = response.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"ok": body.strip() == "ok", "error": body.strip()[:100]}  # incoming webhooks reply with plain "ok"


def send_alert(state: dict, thread_id: str | None = None) -> dict:
    """Post the alert. Returns {"status": "sent" | "skipped" | "failed", "detail": ...}; never raises."""
    token, channel = os.environ.get("SLACK_BOT_TOKEN"), os.environ.get("SLACK_CHANNEL")
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not (token and channel) and not webhook:
        return {"status": "skipped", "detail": "Slack not configured (SLACK_BOT_TOKEN + SLACK_CHANNEL, or SLACK_WEBHOOK_URL)"}
    try:
        text = build_message(state, thread_id)
        if token and channel:
            api = os.environ.get("SLACK_API_BASE", SLACK_API_BASE).rstrip("/")
            reply = _post(f"{api}/chat.postMessage", {"channel": channel, "text": text, "unfurl_links": False}, token)
            target = f"channel {channel}"
        else:
            reply = _post(webhook, {"text": text})
            target = "webhook"
        if not reply.get("ok"):
            return {"status": "failed", "detail": f"Slack error: {reply.get('error', 'unknown')}"}
        return {"status": "sent", "detail": f"posted to {target}"}
    except Exception as exc:  # a Slack problem must never stop the investigation
        return {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"}
