/** Dashboard: what needs attention, and what the system knows about itself.
 *
 *  Ordering is deliberate. "What matters" and "what is due" are the two things
 *  a person opens this app for, so they are first. The derived-store and
 *  deviation panels are last because they are for the person evaluating the
 *  tool, not the person using it.
 */

import { useState } from 'react'
import { api } from '../api'
import { useAsync } from '../lib/hooks'
import { percent, relative } from '../lib/format'
import { Card, Empty, ErrorNote, SourceBadge, Spinner, Stat } from './ui'

export function DashboardView({
  onOpen,
  onOpenPerson,
  onNavigate,
}: {
  onOpen: (docId: string) => void
  onOpenPerson: (personId: string) => void
  onNavigate: (view: string) => void
}) {
  const info = useAsync(() => api.systemInfo(), [])
  const predictions = useAsync(() => api.predictions(), [])
  const insights = useAsync(() => api.insights(), [])
  const people = useAsync(() => api.persons(true), [])
  const timeline = useAsync(() => api.timeline({ limit: 12 }), [])
  const [syncing, setSyncing] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  const docs = info.data?.stores.documents ?? {}
  const total = Object.entries(docs)
    .filter(([name]) => !['search_docs', 'raw_artifacts', 'cursors', 'audit'].includes(name))
    .reduce((sum, [, count]) => sum + count, 0)
  const sources = Object.keys(info.data?.stores.documents ?? {}).filter((name) =>
    ['messages', 'files', 'notes', 'videos', 'transcripts'].includes(name),
  ).length

  async function runSync() {
    setSyncing(true)
    try {
      const result = await api.sync('demo')
      setToast(
        `Synced: ${result.ingest.counters.created ?? 0} new, ` +
          `${result.ingest.counters.updated ?? 0} updated in ${result.took_ms} ms`,
      )
      window.setTimeout(() => setToast(null), 4200)
      info.reload()
      predictions.reload()
      insights.reload()
      people.reload()
      timeline.reload()
    } catch (err) {
      setToast(`Sync failed: ${String(err)}`)
    } finally {
      setSyncing(false)
    }
  }

  if (info.error) return <ErrorNote error={info.error} />
  if (!info.data) return <div className="row"><Spinner /> Loading workspace…</div>

  const deadlines = (predictions.data?.predictions ?? []).filter((p) => p.kind === 'deadline')
  const followUps = (predictions.data?.predictions ?? []).filter(
    (p) => p.kind === 'unanswered_question',
  )

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: '1rem' }}>
        <Stat
          label="Records"
          value={total.toLocaleString()}
          sub={`across ${sources} content sources`}
        />
        <Stat
          label="People"
          value={(people.data?.persons.length ?? 0).toLocaleString()}
          sub="resolved across 2+ sources"
        />
        <Stat
          label="Graph"
          value={Number(info.data.stores.graph.nodes ?? 0).toLocaleString()}
          sub={`${info.data.stores.graph.edges ?? 0} edges`}
        />
        <Stat
          label="Search index"
          value={Number(info.data.search.documents ?? 0).toLocaleString()}
          sub={`${info.data.search.terms ?? 0} terms · ${info.data.search.postings ?? 0} postings`}
        />
      </div>

      <div className="grid grid-2" style={{ marginBottom: '1rem' }}>
        <Card
          title="Deadlines ahead"
          hint="extracted from messages, never inferred"
          actions={
            <button className="btn sm" onClick={() => onNavigate('search')}>
              Search
            </button>
          }
          flush
        >
          {predictions.loading && (
            <div className="card-body row">
              <Spinner /> Loading…
            </div>
          )}
          {!predictions.loading && !deadlines.length && (
            <Empty title="No upcoming deadlines">
              A date is only counted as a deadline when it arrives with a commitment cue — a bare
              “Friday” in a scheduling sentence is not one.
            </Empty>
          )}
          {deadlines.slice(0, 6).map((prediction) => (
            <button
              className="list-item"
              key={prediction._id}
              onClick={() => prediction.source_ids[0] && onOpen(prediction.source_ids[0])}
            >
              <span className="badge warn">{relative(prediction.due)}</span>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div className="truncate">{prediction.title}</div>
                <div className="tiny faint clamp-2">{prediction.detail}</div>
              </div>
              <div className="mono faint">{percent(prediction.confidence)}</div>
            </button>
          ))}
        </Card>

        <Card title="Unanswered questions" hint="asked, then ignored" flush>
          {followUps.length === 0 ? (
            <Empty title="Nothing left hanging">
              A question only counts when enough messages followed it for the silence to mean
              something.
            </Empty>
          ) : (
            followUps.slice(0, 6).map((prediction) => (
              <button
                className="list-item"
                key={prediction._id}
                onClick={() => prediction.source_ids[0] && onOpen(prediction.source_ids[0])}
              >
                <span className="badge danger">?</span>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="truncate">{prediction.title}</div>
                  <div className="tiny faint clamp-2">{prediction.detail}</div>
                </div>
              </button>
            ))
          )}
        </Card>
      </div>

      <div className="grid grid-2" style={{ marginBottom: '1rem' }}>
        <Card
          title="Hidden connections"
          hint="advisory — nothing here is a fact until you accept it"
          flush
        >
          {!insights.data?.insights.length && (
            <Empty title="No connections detected yet">
              Detectors need co-presence data across conversations; run a sync on more sources.
            </Empty>
          )}
          {(insights.data?.insights ?? []).slice(0, 6).map((insight) => (
            <div className="list-item" key={insight._id} style={{ cursor: 'default' }}>
              <span className="badge advisory">{insight.detector}</span>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div className="truncate">{insight.title}</div>
                <div className="tiny faint clamp-2">{insight.detail}</div>
                <div className="row-wrap" style={{ marginTop: '0.2rem' }}>
                  {insight.entities.slice(0, 2).map((entity) =>
                    entity.startsWith('per_') ? (
                      <button
                        key={entity}
                        className="btn sm"
                        onClick={() => onOpenPerson(entity)}
                      >
                        {entity.replace(/^per_/, '').replace(/_/g, ' ')}
                      </button>
                    ) : null,
                  )}
                </div>
              </div>
              <div className="mono faint">{percent(insight.confidence)}</div>
            </div>
          ))}
        </Card>

        <Card
          title="Recent activity"
          hint="every source, one clock"
          actions={
            <button className="btn sm" onClick={() => onNavigate('timeline')}>
              Timeline
            </button>
          }
          flush
        >
          <div className="list">
            {(timeline.data?.events ?? []).slice(0, 7).map((event) => (
              <button className="list-item" key={event.doc_id} onClick={() => onOpen(event.doc_id)}>
                <SourceBadge provider={event.provider} />
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="truncate">{event.title || event.doc_id}</div>
                  <div className="tiny faint">
                    {event.sender_name ? `${event.sender_name} · ` : ''}
                    {relative(event.ts)}
                    {event.deadline_count > 0 ? ` · ${event.deadline_count} deadline` : ''}
                  </div>
                </div>
                {event.importance > 0.55 && <span className="badge accent">important</span>}
              </button>
            ))}
          </div>
        </Card>
      </div>

      <Card
        title="Connect a source"
        hint={`${info.data.mode} mode · ${info.data.workspace}`}
        actions={
          <button className="btn primary" onClick={runSync} disabled={syncing}>
            {syncing ? <Spinner /> : null} Sync demo workspace
          </button>
        }
        flush
      >
        <table className="plain">
          <thead>
            <tr>
              <th>Store</th>
              <th>Documents</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(docs)
              .sort((a, b) => b[1] - a[1])
              .map(([name, count]) => (
                <tr key={name}>
                  <td className="mono">{name}</td>
                  <td className="mono">{count.toLocaleString()}</td>
                  <td className="tiny faint">
                    {name === 'raw_artifacts'
                      ? 'immutable provider payloads, kept verbatim'
                      : name === 'search_docs'
                        ? 'content-free ranking projection — safe to drop'
                        : name === 'cursors'
                          ? 'per-stream sync position'
                          : ''}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </Card>

      <Card
        title="Recorded deviations"
        hint="every place the running system differs from the blueprint, and why"
        style={{ marginTop: '1rem' }}
        flush
      >
        <table className="plain">
          <thead>
            <tr>
              <th style={{ width: 74 }}>ADR</th>
              <th style={{ width: 82 }}>Area</th>
              <th>Blueprint vs. actual, and the reason</th>
            </tr>
          </thead>
          <tbody>
            {info.data.deviations.map((deviation) => (
              <tr key={deviation.id}>
                <td className="mono">{deviation.id.replace('ADR-', '')}</td>
                <td>
                  <span className="badge">{deviation.area}</span>
                </td>
                <td>
                  <div className="row-wrap" style={{ marginBottom: '0.2rem' }}>
                    <span className="tiny faint" style={{ textDecoration: 'line-through' }}>
                      {deviation.blueprint}
                    </span>
                    <span className="tiny">→</span>
                    <span className="tiny" style={{ fontWeight: 560 }}>
                      {deviation.actual}
                    </span>
                  </div>
                  <div className="tiny muted">{deviation.reason}</div>
                  <div className="tiny" style={{ color: 'var(--warn)', marginTop: '0.2rem' }}>
                    Residual risk: {deviation.risk}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {toast && <div className="toast">{toast}</div>}
    </>
  )
}
