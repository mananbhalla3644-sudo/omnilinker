/** Timeline: one clock across every source.
 *
 *  People do not think in "Slack" and "email"; they think in "what happened".
 *  Any UI that partitions by provider makes the user do the joining themselves,
 *  which is exactly the work this product exists to remove. So this view is
 *  chronological, and the provider is a badge rather than a section header.
 */

import { useMemo, useState } from 'react'
import { api } from '../api'
import { useAsync } from '../lib/hooks'
import { dayLabel, relative, shortTime } from '../lib/format'
import { Card, Empty, ErrorNote, SourceBadge, Spinner } from './ui'

const SOURCES = ['slack', 'gmail', 'discord', 'whatsapp', 'gdrive', 'notion', 'youtube'] as const

export function TimelineView({ onOpen }: { onOpen: (docId: string) => void }) {
  const [active, setActive] = useState<string[]>([])
  const [onlyImportant, setOnlyImportant] = useState(false)
  const providers = useMemo(() => active.join(','), [active])
  const { data, error, loading } = useAsync(
    () => api.timeline({ limit: 300, providers: providers || undefined }),
    [providers],
  )

  const events = (data?.events ?? []).filter((e) => !onlyImportant || e.importance > 0.4)

  const days = useMemo(() => {
    const grouped = new Map<string, typeof events>()
    for (const event of events) {
      const day = dayLabel(event.ts)
      const bucket = grouped.get(day)
      if (bucket) bucket.push(event)
      else grouped.set(day, [event])
    }
    return [...grouped.entries()]
  }, [data])

  return (
    <Card
      title="Timeline"
      hint={`${events.length} events`}
      actions={loading ? <Spinner /> : undefined}
    >
      <div className="row-wrap" style={{ marginBottom: '0.9rem' }}>
        {SOURCES.map((source) => (
          <button
            key={source}
            className="chip"
            aria-pressed={active.includes(source)}
            onClick={() =>
              setActive((prev) =>
                prev.includes(source) ? prev.filter((s) => s !== source) : [...prev, source],
              )
            }
          >
            {source}
          </button>
        ))}
        <div className="spacer" />
        <button
          className="chip"
          aria-pressed={onlyImportant}
          onClick={() => setOnlyImportant((v) => !v)}
          title="Ranked by the linear importance model: owner, deadline, decision language, substance, recency"
        >
          important only
        </button>
        {active.length > 0 && (
          <button className="btn sm" onClick={() => setActive([])}>
            Clear
          </button>
        )}
      </div>

      {error && <ErrorNote error={error} />}
      {!loading && !days.length && (
        <Empty title="Nothing on the clock">
          {active.length
            ? 'No events for the selected sources.'
            : 'Run a sync from the dashboard to populate the timeline.'}
        </Empty>
      )}

      <div className="timeline">
        {days.map(([day, bucket]) => (
          <div key={day}>
            <div className="tl-day">
              <span className="tl-day-label">{day}</span>
            </div>
            {bucket.map((event) => (
              <button
                className={`tl-item${event.importance > 0.5 ? ' important' : ''}`}
                key={event.doc_id}
                style={{
                  display: 'flex',
                  gap: '0.6rem',
                  width: '100%',
                  textAlign: 'left',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                }}
                onClick={() => onOpen(event.doc_id)}
              >
                <span className="mono faint" style={{ minWidth: 78, fontSize: 11 }}>
                  {shortTime(event.ts)}
                </span>
                <SourceBadge provider={event.provider} />
                <span style={{ minWidth: 0, flex: 1 }}>
                  <span className="truncate" style={{ display: 'block' }}>
                    {event.title || event.doc_id}
                  </span>
                  <span className="tiny faint">
                    {event.sender_name ? `${event.sender_name} · ` : ''}
                    {event.kind}
                    {event.has_attachment ? ' · attachment' : ''}
                    {event.deadline_count > 0 ? ` · ${event.deadline_count} deadline` : ''}
                  </span>
                </span>
                {event.importance > 0.55 && (
                  <span className="badge accent" title={`importance ${event.importance.toFixed(2)}`}>
                    {relative(event.ts)}
                  </span>
                )}
              </button>
            ))}
          </div>
        ))}
      </div>
    </Card>
  )
}
