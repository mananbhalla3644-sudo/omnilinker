/** Document drawer: the "where did this come from" surface.
 *
 *  Provenance is the point of the product, so this is not an afterthought
 *  modal - it shows lineage, the raw artifact reference, the sealing state, and
 *  the thread siblings next to the body. Every one of those is a question a
 *  user has when they are deciding whether to trust a result.
 */

import { useState } from 'react'
import { api, type Document } from '../api'
import { useAsync } from '../lib/hooks'
import { fullTime, relative } from '../lib/format'
import { Card, Drawer, ErrorNote, SourceBadge, Spinner } from './ui'

function isSealed(value: unknown): boolean {
  return (
    typeof value === 'object' &&
    value !== null &&
    'v' in value &&
    'ct' in value &&
    'nonce' in value
  )
}

function bodyOf(doc: Document): string {
  const candidates = [doc.body_text, doc.text, doc.extracted_text, doc.description]
  for (const value of candidates) {
    if (typeof value === 'string' && value.trim()) return value
  }
  return ''
}

/**
 * A human title for any document.
 *
 * The same fallback chain the search results use, so a message looks the same
 * in a result row and in the drawer. Without the channel/thread fallbacks a
 * Slack message with no subject rendered as its raw document id - technically
 * correct and completely useless to a person trying to recognise what they
 * clicked.
 */
function titleOf(doc: Document): string {
  if (doc.title) return doc.title
  if (doc.name) return doc.name
  const extra = (doc.extra ?? {}) as Record<string, unknown>
  // Real subject first, then the derived thread subject. The channel is a
  // *location*, so it comes last: leading with it made every Slack message
  // read "engineering" in the drawer title.
  if (extra.subject) return String(extra.subject)
  if (doc.thread_subject) return String(doc.thread_subject)
  const location = extra.channel_label ?? extra.channel ?? extra.chat_label
  if (doc.sender_name) {
    return `${doc.sender_name}${location ? ` in ${String(location)}` : ''}`
  }
  if (location) return String(location)
  return doc._id
}

export function DocumentDrawer({ docId, onClose }: { docId: string; onClose: () => void }) {
  const { data, error, loading } = useAsync(() => api.document(docId), [docId])
  const [summary, setSummary] = useState<{ text: string; method: string; compression: number } | null>(
    null,
  )
  const [summarising, setSummarising] = useState(false)
  const [summaryError, setSummaryError] = useState<string | null>(null)

  const doc = data?.document
  const body = doc ? bodyOf(doc) : ''
  const sealed = doc ? isSealed(doc.body_text) : false

  async function runSummary() {
    setSummarising(true)
    setSummaryError(null)
    try {
      const result = await api.summarize(docId)
      setSummary(result)
    } catch (err) {
      setSummaryError(String(err))
    } finally {
      setSummarising(false)
    }
  }

  return (
    <Drawer
      title={doc ? titleOf(doc) : docId}
      subtitle={
        doc ? (
          <span className="row-wrap">
            <SourceBadge provider={String(doc.provider ?? '')} />
            <span>{doc.kind ?? 'message'}</span>
            {doc.ts ? <span>· {fullTime(String(doc.ts))}</span> : null}
          </span>
        ) : undefined
      }
      onClose={onClose}
    >
      {loading && <div className="row"><Spinner /> Loading…</div>}
      {error && <ErrorNote error={error} />}

      {doc && (
        <>
          <Card title="Content" hint={sealed ? 'decrypted on read' : undefined} tight>
            <div className="row-wrap" style={{ marginBottom: '0.5rem' }}>
              <button className="btn sm" onClick={runSummary} disabled={summarising || !body}>
                {summarising ? <Spinner /> : null} Summarize
              </button>
              <span className="tiny faint">
                Extractive — every sentence is a verbatim span of this document.
              </span>
            </div>
            {summaryError && <ErrorNote error={{ detail: summaryError, status: 0 }} />}
            {summary && (
              <div className="evidence" style={{ marginBottom: '0.55rem' }}>
                <div style={{ marginBottom: '0.3rem' }}>{summary.text}</div>
                <div className="tiny faint">
                  {summary.method} · {(summary.compression * 100).toFixed(0)}% of the original
                </div>
              </div>
            )}
            {body ? (
              <div className="body-text">{body}</div>
            ) : (
              <Empty2 text="This record has no text body (file metadata only)." />
            )}
          </Card>

          <Card title="Provenance" hint="blueprint 10.4">
            <dl className="kv">
              <dt>Document id</dt>
              <dd className="mono">{doc._id}</dd>
              <dt>Provider</dt>
              <dd>{String(doc.provider ?? '')}</dd>
              {doc.sender_name ? (
                <>
                  <dt>Sender</dt>
                  <dd>{String(doc.sender_name)}</dd>
                </>
              ) : null}
              {doc.importance !== undefined ? (
                <>
                  <dt>Importance</dt>
                  <dd>{Number(doc.importance).toFixed(3)}</dd>
                </>
              ) : null}
              <dt>Attachments</dt>
              <dd>{(doc.attachments ?? []).length || 'none'}</dd>
              {doc.body_hash ? (
                <>
                  <dt>Content hash</dt>
                  <dd className="mono">{String(doc.body_hash).slice(0, 26)}…</dd>
                </>
              ) : null}
              {doc.lineage ? (
                <>
                  <dt>Raw artifact</dt>
                  <dd className="mono">{String((doc.lineage as Record<string, unknown>).raw_ref ?? '')}</dd>
                  <dt>Ingest run</dt>
                  <dd className="mono">
                    {String((doc.lineage as Record<string, unknown>).ingest_run_id ?? '')}
                  </dd>
                </>
              ) : null}
            </dl>
            <p className="tiny faint" style={{ marginBottom: 0, marginTop: '0.7rem' }}>
              Bodies are stored as AES-256-GCM envelopes bound to{' '}
              <code className="mono">workspace|doc_id|field|schema</code>, so a ciphertext moved
              between documents fails authentication instead of decrypting.
            </p>
          </Card>

          {(doc.attachments ?? []).length > 0 && (
            <Card title="Attachments" flush>
              <div className="list">
                {(doc.attachments ?? []).map((att, index) => (
                  <div className="list-item" key={index} style={{ cursor: 'default' }}>
                    <span className="badge">{att.mime?.split('/').pop() ?? 'file'}</span>
                    <span className="truncate">{att.name ?? 'attachment'}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {(data?.related.length ?? 0) > 0 && (
            <Card title="Same thread" hint={`${data?.related.length} earlier / later`} flush>
              <div className="list">
                {data?.related.map((rel) => (
                  <div className="list-item" key={rel.doc_id} style={{ cursor: 'default' }}>
                    <SourceBadge provider={rel.provider} />
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div className="truncate">{rel.title || '(no subject)'}</div>
                      <div className="tiny faint">{relative(rel.ts)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </Drawer>
  )
}

function Empty2({ text }: { text: string }) {
  return <div className="tiny faint">{text}</div>
}
