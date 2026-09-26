/** API client: the connection lifecycle. */

import { api, type Connector } from './api'

export type Connection = {
  connector_id: string
  connected: boolean
  status: string
  status_detail: string
  connected_at: string
  updated_at: string
  expires_at: string
  needs_refresh: boolean
  scopes: string[]
  access_token_present: boolean
  refresh_token_present: boolean
  provider_context: Record<string, unknown>
  configured: boolean
  credentials_source: string
  auth_flow: string
  redirect_uri: string
  next_step: string
}

export type CredentialReport = {
  connector_id: string
  configured: boolean
  public_client: boolean
  needs_env: boolean
  env_vars: string[]
}

const CONNECTIONS_PATH = '/connections'

export async function listConnections(): Promise<{
  connections: Connection[]
  credentials: CredentialReport[]
}> {
  const response = await fetch(`/api${CONNECTIONS_PATH}`)
  if (!response.ok) throw new Error(`listConnections: ${response.status}`)
  return response.json()
}

export async function beginAuthorize(
  connectorId: string,
): Promise<{ authorization_url: string; state: string; redirect_uri: string }> {
  const response = await fetch(`/api${CONNECTIONS_PATH}/${connectorId}/authorize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(body?.detail?.message ?? body?.detail ?? `authorize failed: ${response.status}`)
  }
  return body
}

export async function disconnect(
  connectorId: string,
): Promise<{ disconnected: boolean; revoked_remotely: boolean; note: string }> {
  const response = await fetch(`/api${CONNECTIONS_PATH}/${connectorId}/disconnect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new Error(`disconnect failed: ${response.status}`)
  return body
}

export async function verifyConnection(
  connectorId: string,
): Promise<{ ok: boolean; streams?: number; sample?: string[]; error?: unknown }> {
  const response = await fetch(`/api${CONNECTIONS_PATH}/${connectorId}/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    return { ok: false, error: body?.error ?? body?.detail ?? response.status }
  }
  return body
}

/**
 * Run the OAuth handshake in a popup.
 *
 * A popup rather than a full-page redirect, so the Connectors view - which
 * holds the filter state, the expanded scope panels, and the connection list -
 * survives the round trip and updates in place when the callback posts back.
 *
 * The interval polls because `postMessage` is the only signal that works when
 * the popup is cross-origin, which it is: the callback is served from our own
 * origin but the *navigation into* it was to the provider's domain, and a popup
 * that was closed without approving never posts anything at all. A timer that
 * notices the popup closing is the only universal completion signal.
 */
export function connectViaPopup(
  connectorId: string,
  onDone: (result: { ok: boolean; message: string }) => void,
): { cancel: () => void } {
  let settled = false
  let popup: Window | null = null

  const finish = (ok: boolean, message: string) => {
    if (settled) return
    settled = true
    window.removeEventListener('message', onMessage)
    if (timer) window.clearInterval(timer)
    onDone({ ok, message })
  }

  function onMessage(event: MessageEvent) {
    const data = event.data
    if (!data || data.type !== 'omni:connection') return
    finish(Boolean(data.ok), String(data.message ?? ''))
  }

  window.addEventListener('message', onMessage)

  beginAuthorize(connectorId)
    .then(({ authorization_url }) => {
      popup = window.open(
        authorization_url,
        `omni-oauth-${connectorId}`,
        'width=620,height=760,menubar=no,toolbar=no,location=yes,status=no',
      )
      if (!popup) {
        finish(
          false,
          'The consent window was blocked. Allow popups for this site, or open the ' +
            'link in a new tab.',
        )
      }
    })
    .catch((err: Error) => finish(false, err.message))

  const timer = window.setInterval(() => {
    if (settled) return
    if (popup && popup.closed) {
      finish(false, 'The consent window was closed before the connection finished.')
    }
  }, 700)

  return {
    cancel: () => {
      popup?.close()
      finish(false, 'Cancelled.')
    },
  }
}

export type { Connector }
export { api }
