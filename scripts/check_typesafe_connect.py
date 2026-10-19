"""Check connectivity to TypeSafe System One (documented API).

Docs:
  https://docs.typesafe.ai/
  https://docs.typesafe.ai/introduction/quickstart
  https://docs.typesafe.ai/api

Official endpoint:
  POST https://api.typesafe.ai/v1/systemone

Usage (from repo root):
  python scripts/check_typesafe_connect.py
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Hard-pin to documented host (never send Bearer key to arbitrary URL).
TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

API_KEY = (
    (os.getenv("TYPESAFE_API_KEY") or "").strip()
    or (os.getenv("JEV_API_KEY") or "").strip()
)

# Minimal Choice request from TypeSafe quickstart / API reference
TYPESAFE_PAYLOAD = {
    "state": (
        "Our API integration started returning 500 errors on every request "
        "about 20 minutes ago, and we can't process any customer orders until this is fixed."
    ),
    "model": "jev-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        }
    },
}

SKY_PAYLOAD = {
    "state": "The sky appears blue on a clear day.",
    "model": "jev-latest",
    "questions": {
        "sky_blue": {
            "type": "noul",
            "instructions": "Is the sky blue?",
        }
    },
}


def _ssl_verify() -> ssl.SSLContext | bool:
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return True


def _probe(name: str, url: str, payload: dict) -> bool:
    print(f"\n=== {name} ===")
    print(f"POST {url}")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0), verify=_ssl_verify()) as client:
            resp = client.post(url, headers=headers, json=payload)
        latency = time.perf_counter() - t0
        print(f"status:  {resp.status_code}")
        print(f"latency: {latency:.3f}s")
        try:
            body = resp.json()
            print("body:")
            print(json.dumps(body, indent=2)[:2000])
        except Exception:
            print("body (raw):")
            print(resp.text[:2000])
        ok = 200 <= resp.status_code < 300
        print("result:", "OK" if ok else "FAILED")
        return ok
    except httpx.TimeoutException as exc:
        print(f"TIMEOUT after {time.perf_counter() - t0:.3f}s: {exc or type(exc).__name__}")
        print("result: FAILED")
        return False
    except httpx.HTTPError as exc:
        detail = str(exc).strip() or repr(exc)
        print(f"TRANSPORT ERROR after {time.perf_counter() - t0:.3f}s: {type(exc).__name__}: {detail}")
        print("result: FAILED")
        return False


def main() -> int:
    print("TypeSafe / Jev connectivity check")
    print("Docs: https://docs.typesafe.ai/  |  https://docs.typesafe.ai/api")
    if not API_KEY:
        print(
            "\nMissing API key. Set TYPESAFE_API_KEY or JEV_API_KEY in .env",
            file=sys.stderr,
        )
        return 1
    print(f"API key: present (len={len(API_KEY)})")
    print(f"SSL verify: {'truststore (OS certs)' if _ssl_verify() is not True else 'default'}")

    ticket_ok = _probe("System One — ticket Choice", TYPESAFE_ENDPOINT, TYPESAFE_PAYLOAD)
    sky_ok = _probe("System One — sky blue Noul", TYPESAFE_ENDPOINT, SKY_PAYLOAD)

    print("\n=== Summary ===")
    print(f"Ticket Choice : {'OK' if ticket_ok else 'FAILED'}")
    print(f"Sky Noul      : {'OK' if sky_ok else 'FAILED'}")
    return 0 if (ticket_ok and sky_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
