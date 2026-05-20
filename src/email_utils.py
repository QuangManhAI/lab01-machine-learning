import hashlib
import html
import logging
import re
from codecs import lookup
from email import policy
from email.parser import BytesParser

logger = logging.getLogger(__name__)


def normalize_label(value):
    text = str(value).strip().lower()
    if text in {"1", "spam", "phishing", "malicious", "bad", "true"}:
        return "spam"
    if text in {"0", "ham", "not spam", "not-spam", "legitimate", "benign", "good", "false"}:
        return "ham"
    if "spam" in text or "phish" in text:
        return "spam"
    return "ham"


def read_body(message):
    parts = message.walk() if message.is_multipart() else [message]
    texts = []
    for part in parts:
        if part.get_content_type() == "text/plain":
            texts.append(decode_text_part(part))
    return "\n".join(texts).strip()


def decode_text_part(part):
    try:
        return part.get_content()
    except LookupError:
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or ""
        logger.warning("Unknown email charset, using fallback decode: %s", charset)
        return decode_bytes(payload, charset)
    except Exception:
        payload = part.get_payload(decode=True) or b""
        logger.warning("Cannot decode email part, using fallback bytes decode", exc_info=True)
        return decode_bytes(payload, part.get_content_charset() or "")


def decode_bytes(payload, charset):
    candidates = [charset, "utf-8", "latin-1"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lookup(candidate)
            return payload.decode(candidate, errors="replace")
        except LookupError:
            continue
    return payload.decode("utf-8", errors="replace")


def email_item(source, source_url, archive_path, raw, label=None):
    message = BytesParser(policy=policy.default).parsebytes(raw)
    body = read_body(message)
    email_id = hashlib.sha256(source_url.encode() + archive_path.encode() + raw).hexdigest()
    return {
        "email_id": email_id,
        "source": source["name"],
        "source_url": source_url,
        "archive_path": archive_path,
        "label": label or source.get("label", "ham"),
        "sender": str(message.get("from", "")),
        "recipient": str(message.get("to", "")),
        "subject": str(message.get("subject", "")),
        "date": str(message.get("date", "")),
        "body": body,
    }


def text_item(source, source_url, archive_path, text, label):
    text = str(text or "").strip()
    raw = text.encode("utf-8", errors="ignore")
    email_id = hashlib.sha256(source_url.encode() + archive_path.encode() + raw).hexdigest()
    return {
        "email_id": email_id,
        "source": source["name"],
        "source_url": source_url,
        "archive_path": archive_path,
        "label": normalize_label(label),
        "sender": "",
        "recipient": "",
        "subject": "",
        "date": "",
        "body": text,
    }


def html_to_email_bytes(text):
    title = first_match(text, r"<title>(.*?)</title>")
    body = re.sub(r"<(script|style).*?</\1>", " ", text, flags=re.I | re.S)
    body = re.sub(r"<[^>]+>", "\n", body)
    body = html.unescape(body)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    body = "\n".join(lines)
    return f"Subject: {title}\n\n{body}".encode("utf-8", errors="ignore")


def first_match(text, pattern):
    match = re.search(pattern, text, flags=re.I | re.S)
    return html.unescape(match.group(1)).strip() if match else ""
