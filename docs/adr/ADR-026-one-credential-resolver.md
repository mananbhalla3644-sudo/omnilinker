# ADR-026: One credential resolver for the whole OAuth handshake

**Status:** accepted
**Area:** connectors / OAuth

## The decision

Every step of an OAuth handshake resolves the provider's client credentials
through exactly one function, `connections.client_credentials(provider)`, which
reads the `OMNI_OAUTH_<PROVIDER>_CLIENT_ID` / `_CLIENT_SECRET` convention.

`BaseRestConnector._credentials()` is the only place the REST connectors obtain
those values, and `authorize`, `exchange_code` and `refresh_tokens` all call it.

## What went wrong

The two halves of the handshake originally resolved credentials independently:

| step | how it read credentials |
|---|---|
| `authorize` | `connections.client_credentials()` → `OMNI_OAUTH_SLACK_CLIENT_ID` |
| `exchange_code` | `os.environ.get("SLACK_CLIENT_ID", "demo-client-id")` |
| `refresh_tokens` | `os.environ.get("SLACK_CLIENT_ID", "demo-client-id")` |

The unprefixed name and the literal `"demo-client-id"` fallback belong to the
demo connector's original scaffolding. They were never updated when the
`OMNI_OAUTH_` convention was introduced.

The result was a failure that pointed at the wrong thing. The app reported
itself correctly configured, the browser was sent to Slack's approval screen
with the real Client ID, the user approved, and then the single step that
actually authenticates the client sent the literal string `demo-client-id` and
received `invalid_client`. The natural reading of that error is "the secret I
pasted is wrong" — and the user's credentials had never been read at all.

Two independent things had to be true for this to survive: the placeholder only
appears when the prefixed variable is *absent*, so it looks like a
configuration problem rather than a code path; and the authorize step still
works, so the flow gets all the way to the approval screen before failing.

## Why one resolver

Two steps that disagree about who the client is cannot both be right, and the
failure surfaces at whichever step runs last. Resolving once means the value
shown in the UI, sent to the provider for approval, and presented at the token
exchange are the same string by construction rather than by inspection.

The convention is `OMNI_OAUTH_<PROVIDER>_*` rather than `<PROVIDER>_*` because
the unprefixed form collides with unrelated environment variables, and because a
prefix makes `env | grep OMNI_OAUTH` a complete inventory of configured
providers. Public clients (those the provider issues no secret for) may carry
`public_client_id` in their descriptor, since such an identifier is published in
the provider's own documentation and is not a secret.

## The error message is part of the decision

A bare `invalid_client` is not actionable, so token-endpoint failures are
translated by `_explain_exchange_failure` into the specific likely cause. For
`invalid_client` it says, explicitly, that the value may be the Verification
Token or Signing Secret rather than the Client Secret, because those are the
three values printed next to each other on every provider console. The same
applies to `redirect_uri_mismatch`, `invalid_scope` and `invalid_code`.

## Rejected: probing the secret in isolation

`POST /api/connections/{id}/diagnose` accepts a candidate secret and reports
whether the provider accepts it, which would settle the Client ID / Client
Secret / Verification Token ambiguity without a browser.

It cannot work for Slack. `oauth.v2.access` validates the authorization code
before the client credentials, so a correct secret and a nonsense one both
return `invalid_code`. Two earlier versions of the probe reported success for a
deliberately wrong value.

The endpoint therefore runs a **control** request with a known-bad secret
first, and reports `inconclusive` whenever the control and the candidate agree.
It only claims `credentials_accepted` when the provider actually distinguished
the two. The value is never persisted. Reporting nothing is strictly better
than reporting a guess, because a user who is told "your secret is fine" stops
looking.

## Consequence

Adding a provider requires no change to any credential-handling code. A
provider whose secret cannot be tested in isolation is not special-cased; the
control detects it automatically and defers to the real handshake.
