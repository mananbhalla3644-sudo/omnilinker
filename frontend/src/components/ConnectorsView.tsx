/** Connectors + the data passport.
 *
 *  Two things a user must be able to check before connecting anything:
 *
 *  1. **Exactly what will be read.** Every scope carries a justification, and a
 *     connector will not register without one (blueprint 6.2.1, check C1). A
 *     permission list with no stated reason is not consent, it is a prompt.
 *
 *  2. **What will happen to the data.** The passport answers "what do you have
 *     on me, from where, sealed how, and can you write back?" from the data
 *     itself, not from configuration - so it cannot drift from reality.
 *
 *  The Connect button is a real OAuth handshake: it mints a PKCE pair
 *  server-side, opens the provider's consent screen in a popup, and stores a
 *  *sealed* refresh token when the callback returns. No token ever reaches this
 *  component - the status endpoint reports presence, not value.
 */

import { useCallback, useEffect, useState } from 'react'
import { api, type Connector } from '../api'
import {
  connectViaPopup,
  disconnect,
  listConnections,
  verifyConnection,
  type Connection,
  type CredentialReport,
} from '../connections'
import { useAsync } from '../lib/hooks'
import { fullTime } from '../lib/format'
import { Card, Empty, ErrorNote, SourceBadge, Spinner } from './ui'

const SENSITIVITY: Record<string, string> = {
  normal: '',
  sensitive: 'warn',
  destructive: 'danger',
}

type Phase = 'idle' | 'authorizing' | 'verifying' | 'syncing'

export function ConnectorsView({ onToast }: { onToast: (message: string) => void }) {
  const connectors = useAsync(() => api.connectors(), [])
  const passport = useAsync(() => api.passport(), [])
  const [connections, setConnections] = useState<Connection[]>([])
  const [credentials, setCredentials] = useState<CredentialReport[]>([])
  const [phase, setPhase] = useState<Record<string, Phase>>({})
  const [notices, setNotices] = useState<Record<string, string>>({})

  const refreshConnections = useCallback(async () => {
    try {
      const data = await listConnections()
      setConnections(data.connections)
      setCredentials(data.credentials)
    } catch {
      // A failed status fetch must not blank the connector list; the buttons
      // fall back to "connect" and the user can retry.
    }
  }, [])

  useEffect(() => {
    void refreshConnections()
  }, [refreshConnections])

  const list = connectors.data?.connectors ?? []
  const data = passport.data?.passport as Record<string, unknown> | undefined
  const keystore = (data?.keystore ?? {}) as Record<string, string>
  const byId = new Map(connections.map((c) => [c.connector_id, c]))
  const credById = new Map(credentials.map((c) => [c.connector_id, c]))

  function setState(id: string, next: Phase) {
    setPhase((prev) => ({ ...prev, [id]: next }))
  }

  function notice(id: string, text: string) {
    setNotices((prev) => ({ ...prev, [id]: text }))
  }

  function connect(connector: Connector) {
    const id = connector.id
    setState(id, 'authorizing')
    notice(id, '')
    connectViaPopup(id, ({ ok, message }) => {
      setState(id, 'idle')
      if (ok) {
        onToast(`${id} connected. ${message}`)
        void refreshConnections()
      } else {
        notice(id, message)
      }
    })
  }

  async function disconnectConnector(id: string) {
    setState(id, 'verifying')
    try {
      const result = await disconnect(id)
      onToast(
        result.disconnected
          ? `${id} disconnected${result.revoked_remotely ? ' and revoked at the provider' : ''}${
              result.note ? ` (${result.note})` : ''
            }`
          : `${id} was not connected`,
      )
      await refreshConnections()
    } catch (err) {
      onToast(`${id}: ${String(err)}`)
    } finally {
      setState(id, 'idle')
    }
  }

  async function verify(id: string) {
    setState(id, 'verifying')
    try {
      const result = await verifyConnection(id)
      if (result.ok) onToast(`${id}: grant works, ${result.streams} streams available`)
      else {
        const detail =
          typeof result.error === 'object' && result.error
            ? JSON.stringify(result.error).slice(0, 200)
            : String(result.error)
        notice(id, `Grant rejected: ${detail}`)
        await refreshConnections()
      }
    } finally {
      setState(id, 'idle')
    }
  }

  async function syncNow(id: string) {
    setState(id, 'syncing')
    try {
      const result = await api.sync(id)
      const c = result.ingest.counters ?? {}
      onToast(
        `${id}: ${c.created ?? 0} new, ${c.updated ?? 0} updated, ${c.failed ?? 0} failed` +
          ` (${result.took_ms} ms)`,
      )
      await refreshConnections()
    } catch (err) {
      notice(id, String(err))
    } finally {
      setState(id, 'idle')
    }
  }

  return (
    <>
      <Card
        title="Connectors"
        hint="registration is gated: scopes need justifications, rate limits must be declared, normalizers must be pure"
        actions={connectors.loading ? <Spinner /> : undefined}
        flush
      >
        {connectors.error && <ErrorNote error={connectors.error} />}
        <div className="list">
          {list.map((connector) => (
            <ConnectorRow
              key={connector.id}
              connector={connector}
              connection={byId.get(connector.id)}
              credential={credById.get(connector.id)}
              phase={phase[connector.id] ?? 'idle'}
              notice={notices[connector.id] ?? ''}
              onConnect={() => connect(connector)}
              onDisconnect={() => void disconnectConnector(connector.id)}
              onVerify={() => void verify(connector.id)}
              onSync={() => void syncNow(connector.id)}
            />
          ))}
          {!connectors.loading && !list.length && <Empty title="No connectors registered" />}
        </div>
      </Card>

      <div className="grid grid-2" style={{ marginTop: '1rem' }}>
        <Card title="Data passport" hint="answered from the data, not from config">
          {passport.error && <ErrorNote error={passport.error} />}
          {!data && (
            <div className="row">
              <Spinner /> Loading…
            </div>
          )}
          {data && (
            <div className="col">
              <dl className="kv">
                <dt>Workspace</dt>
                <dd className="mono">{String(data.workspace)}</dd>
                <dt>Mode</dt>
                <dd>{String(data.mode)}</dd>
                <dt>Key fingerprint</dt>
                <dd className="mono">{String(data.key_fingerprint)}</dd>
                <dt>Identities held</dt>
                <dd>{String(data.identities_held)}</dd>
                <dt>Raw artifacts</dt>
                <dd>{String(data.raw_artifacts_retained)} (sealed provider payloads)</dd>
                <dt>Write-back</dt>
                <dd>
                  <span className="badge ok">disabled</span>{' '}
                  <span className="tiny faint">
                    {(data.write_back as Record<string, string>).reason}
                  </span>
                </dd>
              </dl>

              <div>
                <div className="stat-label" style={{ marginBottom: '0.25rem' }}>
                  By source
                </div>
                <table className="plain">
                  <thead>
                    <tr>
                      <th>Provider</th>
                      <th style={{ textAlign: 'right' }}>Records</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      (data.sources as Record<string, Record<string, number>>) ?? {},
                    ).map(([provider, buckets]) => (
                      <tr key={provider}>
                        <td>
                          <SourceBadge provider={provider} />
                        </td>
                        <td className="mono" style={{ textAlign: 'right' }}>
                          {Object.values(buckets).reduce((a, b) => a + b, 0)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="tiny faint" style={{ marginBottom: 0 }}>
                Every derived store — search projection, persons, insights, predictions — is
                droppable and rebuildable from the source documents. The passport is therefore an
                operation rather than a promise: the API can destroy them and reconstruct them,
                and the records do not move.
              </p>
            </div>
          )}
        </Card>

        <Card title="Keystore" hint="what holds the keys, and what does not">
          {data && (
            <div className="col">
              <dl className="kv">
                <dt>Scheme</dt>
                <dd className="mono">{keystore.scheme}</dd>
                <dt>Workspace</dt>
                <dd className="mono">{keystore.workspace}</dd>
                <dt>Location</dt>
                <dd className="mono">{keystore.keystore}</dd>
                <dt>Binding</dt>
                <dd>{keystore.kdf}</dd>
                <dt>Reconciled</dt>
                <dd>{fullTime(String(data.generated_at))}</dd>
              </dl>
              <div className="evidence">
                <div className="tiny" style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                  Key hierarchy
                </div>
                <pre className="mono tiny" style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
{`master key (UMK, file, 0600)
  └─ workspace KEK        = HKDF(UMK, "wkek|ws")
       ├─ per-source DEK   = HKDF(WKEK, "dek|<source>")
       │                    encrypts content
       ├─ identity pepper  = HKDF(WKEK, "identity-pepper|ws")
       │                    derives comparable identity tokens
       └─ credential key   = HKDF(WKEK, "credential|<provider>")
                            encrypts OAuth tokens`}
                </pre>
              </div>
              <p className="tiny faint" style={{ marginBottom: 0 }}>
                Per-source DEKs never encrypt each other&apos;s content, and credential keys are
                separate again — so a leaked content key cannot be used to read a refresh token,
                and rotating one does not strand the other. The identity pepper is workspace-wide
                on purpose: cross-source resolution is impossible if tokens differ by source.
              </p>
            </div>
          )}
        </Card>
      </div>
    </>
  )
}

function ConnectorRow({
  connector,
  connection,
  credential,
  phase,
  notice,
  onConnect,
  onDisconnect,
  onVerify,
  onSync,
}: {
  connector: Connector
  connection?: Connection
  credential?: CredentialReport
  phase: Phase
  notice: string
  onConnect: () => void
  onDisconnect: () => void
  onVerify: () => void
  onSync: () => void
}) {
  const [open, setOpen] = useState(false)
  const busy = phase !== 'idle'
  const connected = Boolean(connection?.connected)
  const isDemo = connector.id === 'demo'
  const needsEnv = Boolean(credential?.needs_env) && !isDemo
  // A Connect button is only meaningful for a real consent screen. Offering one
  // for a file-export connector is a dead end dressed as a feature.
  const isOAuth = connector.auth_flow === 'oauth2_code' && Boolean(connector.authorize_url)
  const isFileBased = connector.auth_flow === 'export_file' || connector.auth_flow === 'file'
  const needsNothing = connector.auth_flow === 'none'

  return (
    <div style={{ borderBottom: '1px solid var(--border)' }}>
      <div className="list-item" style={{ cursor: 'default' }}>
        <SourceBadge provider={connector.id} label={connector.id} />
        <div style={{ minWidth: 0, flex: 1 }}>
          <div className="row-wrap">
            <span style={{ fontWeight: 560 }}>{connector.display_name}</span>
            <span className="badge">{connector.auth_flow}</span>
            {connected ? (
              <span className="badge ok">connected</span>
            ) : needsNothing || isFileBased ? (
              <span className="badge ok">no account needed</span>
            ) : needsEnv ? (
              <span className="badge danger">needs credentials</span>
            ) : connection?.status === 'refresh_failed' || connection?.status === 'unauthorized' ? (
              <span className="badge danger">{connection.status.replace('_', ' ')}</span>
            ) : (
              <span className="badge warn">not connected</span>
            )}
            {connector.realtime_webhooks && <span className="badge">realtime</span>}
            {connector.incremental_cursor && <span className="badge">incremental</span>}
            {connector.supports_tombstones && <span className="badge">deletions</span>}
          </div>
          <div className="tiny faint" style={{ marginTop: '0.15rem' }}>
            {connector.content_kinds.join(' · ')} · {connector.rate_limit_per_sec}/s
          </div>
          {connected && connection?.scopes?.length ? (
            <div className="tiny faint" style={{ marginTop: '0.2rem' }}>
              granted: {connection.scopes.join(' ')}
              {connection.expires_at ? ` · token expires ${fullTime(connection.expires_at)}` : ''}
              {connection.needs_refresh ? ' · refreshes on next use' : ''}
            </div>
          ) : (
            <div className="tiny faint" style={{ marginTop: '0.2rem' }}>
              {connection?.next_step ?? (isDemo ? 'Synthetic dataset. Sync to populate everything.' : '')}
            </div>
          )}
          {needsEnv && credential?.env_vars?.length ? (
            <div className="tiny" style={{ marginTop: '0.3rem' }}>
              {credential.env_vars.map((v) => (
                <code className="mono" key={v} style={{ marginRight: '0.5rem' }}>
                  {v}
                </code>
              ))}
            </div>
          ) : null}
          {notice ? (
            <div className="tiny" style={{ marginTop: '0.3rem', color: 'var(--danger)' }}>
              {notice}
            </div>
          ) : null}
        </div>

        <div className="row" style={{ flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <button className="btn sm" onClick={() => setOpen((v) => !v)}>
            {open ? 'Less' : 'Scopes'}
          </button>
          {isDemo ? (
            <button className="btn primary sm" onClick={onSync} disabled={busy}>
              {phase === 'syncing' ? <Spinner /> : null} Sync
            </button>
          ) : connected ? (
            <>
              <button
                className="btn sm"
                onClick={onVerify}
                disabled={busy}
                title="Call the provider and confirm the grant still works"
              >
                {phase === 'verifying' ? <Spinner /> : null} Verify
              </button>
              <button className="btn sm" onClick={onSync} disabled={busy}>
                {phase === 'syncing' ? <Spinner /> : null} Sync
              </button>
              <button className="btn sm" onClick={onDisconnect} disabled={busy}>
                Disconnect
              </button>
            </>
          ) : isOAuth ? (
            <button
              className="btn primary sm"
              onClick={onConnect}
              disabled={busy || needsEnv}
              title={
                needsEnv
                  ? 'Set the client credentials shown above, then restart'
                  : 'Open the provider consent screen'
              }
            >
              {phase === 'authorizing' ? <Spinner /> : null} Connect
            </button>
          ) : (
            <span
              className="badge"
              title={
                isFileBased
                  ? 'This connector reads a file you export from the provider'
                  : 'This connector is configured with credentials, not a consent screen'
              }
            >
              {isFileBased ? 'file import' : 'credentials'}
            </span>
          )}
        </div>
      </div>

      {open && (
        <div className="card-body" style={{ background: 'var(--surface-2)' }}>
          <div className="stat-label" style={{ marginBottom: '0.3rem' }}>
            Scopes and why each is needed
          </div>
          <div className="col" style={{ gap: '0.3rem' }}>
            {connector.scopes.map((scope) => {
              const granted = connection?.scopes?.includes(scope.name)
              return (
                <div className="row tiny" key={scope.name}>
                  <span className={`badge ${granted ? 'ok' : ''}`}>{scope.name}</span>
                  <span className={`badge ${SENSITIVITY[scope.sensitivity] ?? ''}`}>
                    {scope.sensitivity}
                  </span>
                  <span className="muted">{scope.justification}</span>
                </div>
              )
            })}
          </div>
          {isOAuth && connection?.redirect_uri ? (
            <p className="tiny faint" style={{ marginTop: '0.6rem', marginBottom: 0 }}>
              Redirect URI to register with the provider:{' '}
              <code className="mono">{connection.redirect_uri}</code>
            </p>
          ) : null}
          <p className="tiny faint" style={{ marginTop: '0.6rem', marginBottom: 0 }}>
            {connector.notes}
          </p>
        </div>
      )}
    </div>
  )
}
