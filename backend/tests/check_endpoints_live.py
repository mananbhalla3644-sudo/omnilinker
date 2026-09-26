"""Live reachability check for every connector's OAuth endpoints.

Run explicitly, never as part of the suite:

    python -m tests.check_endpoints_live

The point is that a wrong `authorize_url` is invisible to every other test. The
app reports the connector as correctly configured, the Connect button opens a
window, and the provider answers with an error page that never comes back to us
- so nothing in our logs, no exception, no failed assertion. Slack's authorize
URL was `https://slack.com/api/oauth.v2/authorize` (wrong host path and a slash
where a dot belonged) and the Connect button was dead for exactly that reason.

The suite cannot catch this because reaching the internet is not a unit-test
concern. This script can, and its findings are pasted into the code.
"""

from __future__ import annotations

import sys

import httpx

from omnilinker.connectors.registry import get_registry, register_builtins

UA = {"User-Agent": "OmniLinker endpoint check (diagnostic)"}

# What each provider's *authorize* page should look like when it is real.
# The failure this catches is subtle: a wrong endpoint does not return an error
# status, it returns an error *page*, so only the body distinguishes them.
EXPECTED_AUTHORIZE_MARKERS = {
    "slack": ("workspace-signin", "signin", "oauth", "log in", "sign in"),
    "discord": ("discord", "login", "oauth2", "authorize"),
    "notion": ("notion", "login", "oauth"),
    "evernote": ("evernote", "login", "oauth"),
    "gdrive": ("google", "accounts", "signin", "serviceaccounts"),
    "gmail": ("google", "accounts", "signin", "oauth"),
    "onedrive": ("microsoft", "login", "oauth2"),
    "dropbox": ("dropbox", "login", "oauth2"),
    "youtube": ("google", "accounts", "signin", "oauth"),
}

# Providers whose token endpoint answers a bad request with an HTML page rather
# than JSON. Evernote's real token endpoint returns a page titled "Evernote
# Error" - so demanding JSON flags a *correct* URL as wrong, which is how a
# diagnostic ends up lying. The marker is what proves the host is Evernote's.
EXPECTED_TOKEN_HTML_MARKERS = {
    "evernote": ("evernote error", "evernote"),
}


def probe_authorize(connector_id: str, url: str) -> tuple[str, str]:
    """Follow the authorize URL with a harmless scope and no real credentials.

    A real provider either shows a consent page or redirects to a sign-in page.
    A wrong endpoint returns a generic error page or a JSON `unknown_method`,
    which is what we are looking for.
    """
    probe = url + ("&" if "?" in url else "?") + "client_id=0000000000.0000000000&scope=x&state=probe"
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(probe, headers=UA)
    except Exception as exc:
        return "unreachable", f"{type(exc).__name__}: {exc}"[:120]

    body = response.text.lower()
    if "unknown_method" in body or "unknown method" in body:
        return "WRONG-ENDPOINT", "provider says this method does not exist"
    if "there's been a glitch" in body or "there has been a glitch" in body:
        return "WRONG-ENDPOINT", "provider's generic error page"
    if any(marker in body for marker in EXPECTED_AUTHORIZE_MARKERS.get(connector_id, ())):
        return "ok", f"HTTP {response.status_code}, recognised provider page"
    # A structured JSON error means the endpoint parsed the request and rejected
    # it, which only a real endpoint does. Notion validates client_id exactly
    # this way, so treating it as a failure would flag a working URL.
    if "json" in response.headers.get("content-type", ""):
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            return "ok", f"HTTP {response.status_code}, rejected as expected: {payload['error']}"
    return "WRONG-ENDPOINT", f"HTTP {response.status_code}, no marker matched"


def probe_token(connector_id: str, url: str) -> tuple[str, str]:
    """A token endpoint must reject a bogus grant, not 404.

    `invalid_grant` / `invalid_client` means the endpoint exists and is
    answering. A 404 or an HTML body means the URL is wrong.
    """
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                url,
                data={"grant_type": "authorization_code", "code": "probe"},
                headers=UA,
            )
    except Exception as exc:
        return "unreachable", f"{type(exc).__name__}: {exc}"[:120]

    content_type = response.headers.get("content-type", "")
    body = response.text
    if "json" not in content_type:
        # An HTML body is only acceptable if it is *this provider's* error page.
        # Otherwise a URL that lands on some CDN's HTML looks identical to a
        # working endpoint, and the check would pass a broken connector.
        for marker in EXPECTED_TOKEN_HTML_MARKERS.get(connector_id, ()):
            if marker in body.lower():
                return "ok", f"HTTP {response.status_code}, provider's own HTML error page"
        return ("WRONG-ENDPOINT",
                f"HTTP {response.status_code}, {content_type or 'no content-type'}, "
                f"no provider marker in body")
    try:
        payload = response.json()
    except ValueError:
        return "WRONG-ENDPOINT", "advertised JSON but the body did not parse"
    if isinstance(payload, dict) and payload.get("error"):
        # A structured rejection means the endpoint exists and is validating the
        # request. Notion answers a bogus client_id exactly this way, which is
        # correct behaviour and must not be reported as a broken URL.
        return "ok", f"rejected as expected: {payload['error']}"
    return "unknown", f"HTTP {response.status_code}, no error field: {str(payload)[:80]}"


def main() -> int:
    register_builtins()
    registry = get_registry()
    print("This makes real network requests to provider endpoints.\n")
    bad = 0
    for connector_id in sorted(registry.ids()):
        descriptor = registry.get(connector_id).descriptor
        if descriptor.auth_flow != "oauth2_code":
            print(f"  {connector_id:10} {descriptor.auth_flow:14} (not an OAuth connector)")
            continue
        print(f"  {connector_id}")
        for label, url, probe in (
            ("authorize", descriptor.authorize_url, probe_authorize),
            ("token", descriptor.token_url, probe_token),
        ):
            if not url:
                print(f"    {label:9} MISSING")
                bad += 1
                continue
            verdict, detail = probe(connector_id, url)
            flag = "  " if verdict == "ok" else "!!"
            if verdict.startswith("WRONG"):
                bad += 1
            print(f"    {flag} {label:9} {verdict:16} {url}")
            print(f"       {'':9} {detail}")
        for label, url in (("refresh", descriptor.refresh_url),):
            if url:
                print(f"       {label:9} {url}")
    print()
    if bad:
        print(f"{bad} endpoint(s) look wrong.")
        return 1
    print("all endpoints look real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
