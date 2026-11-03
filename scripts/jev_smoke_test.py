"""Optional live smoke test for TypeSafe System One API (uses JEV_API_KEY from .env).

Docs: https://docs.typesafe.ai/api
Endpoint: POST {JEV_BASE_URL}/v1/systemone
"""

from __future__ import annotations

import json
import os
import ssl
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

KEY = (os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY") or "").strip()

if not KEY:
    print("Set JEV_API_KEY in .env", file=sys.stderr)
    sys.exit(1)

# Hard-pin to documented host (never send Bearer key to arbitrary URL).
url = "https://api.typesafe.ai/v1/systemone"

payload = {
    "state": {
        "query": "Could a state prisoner obtain federal habeas relief based on newly discovered evidence?",
        "candidate": """
        Claims of newly discovered evidence do not, standing alone,
        provide a basis for federal habeas corpus relief absent an
        independent constitutional violation.
        """
    },
    "model": "jev-latest",
    "questions": {
        "relevance_score": {
            "type": "noul",
            "instructions": """
            Does the candidate passage directly answer or support the query?

            True if:
            - The passage contains legal reasoning relevant to the query.
            - The passage could reasonably be cited as support for the query.
            - The key issue discussed in the query is addressed.

            False if:
            - The passage is only tangentially related.
            - The passage discusses a different legal issue.
            - The passage lacks enough information to support the query.
            """
        }
    }
}
try:
    import truststore

    verify: ssl.SSLContext | bool = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
except Exception:
    verify = True

print(f"POST {url}")
with httpx.Client(timeout=60.0, verify=verify) as client:
    resp = client.post(
        url,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        json=payload,
    )
print(resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2)[:2000])
except Exception:
    print(resp.text[:2000])
sys.exit(0 if resp.status_code < 400 else 1)
