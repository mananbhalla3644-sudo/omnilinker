/** People: resolved identities, with the evidence for every merge.
 *
 *  The evidence panel is the feature. Anyone can show a list of people; the
 *  thing that makes an automatic merge defensible is showing the user *why*
 *  the system thinks two accounts are the same person, and giving them a way to
 *  disagree. Anything less and the resolution is a leap of faith.
 */

import { useState } from 'react'
import { api, type Person } from '../api'
import { useAsync } from '../lib/hooks'
import { relative } from '../lib/format'
import { Card, Empty, ErrorNote, SourceBadge, Spinner } from './ui'

export function PeopleView({ onOpen }: { onOpen: (docId: string) => void }) {
  const [crossOnly, setCrossOnly] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)
  const list = useAsync(() => api.persons(crossOnly), [crossOnly])
  const suggestions = useAsync(() => api.suggestions(), [])

  return (
    <>
      <div className="grid" style={{ gridTemplateColumns: 'minmax(0,340px) minmax(0,1fr)' }}>
        <Card
          title="People"
          hint={`${list.data?.count ?? 0} resolved`}
          actions={
            <button
              className="chip"
              aria-pressed={crossOnly}
              onClick={() => setCrossOnly((v) => !v)}
            >
              cross-source only
            </button>
          }
          flush
        >
          {list.error && <ErrorNote error={list.error} />}
          {list.loading && (
            <div className="card-body row">
              <Spinner /> Loading…
            </div>
          )}
          <div className="list">
            {(list.data?.persons ?? []).map((person) => (
              <PersonRow
                key={person.person_id}
                person={person}
                active={selected === person.person_id}
                onSelect={() => setSelected(person.person_id)}
              />
            ))}
          </div>
        </Card>

        <div className="col">
          {selected ? (
            <PersonDetail id={selected} onOpen={onOpen} />
          ) : (
            <Card title="Identity resolution">
              <Empty title="Pick someone">
                A person is a cluster of per-source accounts. Two accounts merge only on
                decisive shared evidence — an identical deterministic email token, or a phone
                number plus a name. Everything else becomes a suggestion for you to accept.
              </Empty>
              <div className="grid grid-2" style={{ marginTop: '1rem' }}>
                <div>
                  <div className="stat-label" style={{ marginBottom: '0.3rem' }}>
                    Feature weights
                  </div>
                  <dl className="kv">
                    <dt>email token</dt>
                    <dd className="mono">1.00 decisive</dd>
                    <dt>phone token</dt>
                    <dd className="mono">0.85</dd>
                    <dt>first name</dt>
                    <dd className="mono">0.60</dd>
                    <dt>same handle</dt>
                    <dd className="mono">0.55</dd>
                    <dt>co-occurred</dt>
                    <dd className="mono">0.30</dd>
                    <dt>shared thread</dt>
                    <dd className="mono">0.25</dd>
                    <dt>org domain</dt>
                    <dd className="mono">0.15</dd>
                  </dl>
                </div>
                <div>
                  <div className="stat-label" style={{ marginBottom: '0.3rem' }}>
                    Decision bands
                  </div>
                  <dl className="kv">
                    <dt>≥ 0.90</dt>
                    <dd>auto-merge</dd>
                    <dt>0.65–0.90</dt>
                    <dd>suggest, pending your confirmation</dd>
                    <dt>&lt; 0.65</dt>
                    <dd>distinct people</dd>
                  </dl>
                  <p className="tiny faint" style={{ marginTop: '0.6rem' }}>
                    Positive evidence saturates at 1.0, so several weak signals cannot add up to
                    a confident merge. Only a decisive feature crosses the bar on its own.
                  </p>
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>

      <Card
        title="Pending link suggestions"
        hint="the system is not allowed to merge these on its own"
        style={{ marginTop: '1rem' }}
        flush
      >
        {!suggestions.data?.suggestions.length && (
          <Empty title="No suggestions waiting">
            Every cross-source pair scored either above the merge threshold or below the suggest
            threshold.
          </Empty>
        )}
        {(suggestions.data?.suggestions ?? []).map((suggestion) => (
          <div className="list-item" key={suggestion._id} style={{ cursor: 'default' }}>
            <span className="badge advisory">{suggestion.score.toFixed(2)}</span>
            <div style={{ minWidth: 0, flex: 1 }}>
              <div className="row-wrap">
                <strong>{suggestion.display_name_a}</strong>
                <SourceBadge provider={suggestion.provider_a} />
                <span className="faint">↔</span>
                <strong>{suggestion.display_name_b}</strong>
                <SourceBadge provider={suggestion.provider_b} />
              </div>
              <div className="row-wrap" style={{ marginTop: '0.25rem' }}>
                {suggestion.evidence.map((item) => (
                  <span className="badge ok" key={item.feature} title={item.detail}>
                    {item.feature} +{item.weight}
                  </span>
                ))}
                {suggestion.conflicts.map((item) => (
                  <span className="badge danger" key={item.feature} title={item.detail}>
                    {item.feature} {item.weight}
                  </span>
                ))}
              </div>
            </div>
            <span className="badge warn">awaiting you</span>
          </div>
        ))}
      </Card>
    </>
  )
}

function PersonRow({
  person,
  active,
  onSelect,
}: {
  person: Person
  active: boolean
  onSelect: () => void
}) {
  return (
    <button className="list-item" aria-selected={active} onClick={onSelect}>
      <span
        className="badge accent"
        style={{ minWidth: 30, justifyContent: 'center', fontSize: 11 }}
      >
        {person.display_name.slice(0, 2).toUpperCase()}
      </span>
      <div style={{ minWidth: 0, flex: 1 }}>
        <div className="row-wrap">
          <span style={{ fontWeight: 560 }}>{person.display_name}</span>
          {person.cross_source && <span className="badge ok">merged</span>}
        </div>
        <div className="tiny faint truncate">
          {person.providers.join(' · ')} · {person.message_count} records
        </div>
      </div>
    </button>
  )
}

function PersonDetail({ id, onOpen }: { id: string; onOpen: (docId: string) => void }) {
  const { data, error, loading } = useAsync(() => api.person(id), [id])
  if (loading) return <Card><div className="row"><Spinner /> Loading…</div></Card>
  if (error) return <Card><ErrorNote error={error} /></Card>
  if (!data) return null

  return (
    <>
      <Card
        title={data.person.display_name}
        hint={`${data.person.providers.length} sources · ${data.person.identity_count} accounts`}
        actions={
          <div className="row-wrap">
            {data.person.providers.map((provider) => (
              <SourceBadge key={provider} provider={provider} />
            ))}
          </div>
        }
      >
        <div className="row-wrap" style={{ marginBottom: '0.8rem' }}>
          {data.person.aliases.map((alias) => (
            <span className="badge warn" key={alias}>
              aka {alias}
            </span>
          ))}
          <span className="badge">{data.person.message_count} records</span>
          <span className="badge">{data.person.contact_count} known contacts</span>
          {data.person.last_seen && (
            <span className="badge">last seen {relative(data.person.last_seen)}</span>
          )}
        </div>

        <div className="stat-label" style={{ marginBottom: '0.3rem' }}>
          Accounts merged into this person
        </div>
        <table className="plain">
          <thead>
            <tr>
              <th>Source</th>
              <th>Name as that source knows it</th>
              <th>Contact</th>
              <th style={{ textAlign: 'right' }}>Mentions</th>
            </tr>
          </thead>
          <tbody>
            {data.identities.map((identity) => (
              <tr key={identity.identity_id}>
                <td>
                  <SourceBadge provider={identity.provider} />
                </td>
                <td>{identity.display_name || '—'}</td>
                <td className="tiny">
                  {identity.email_present ? <span className="badge ok">email</span> : null}{' '}
                  {identity.phone_present ? <span className="badge ok">phone</span> : null}{' '}
                  {!identity.email_present && !identity.phone_present ? (
                    <span className="faint">name only</span>
                  ) : null}
                </td>
                <td className="mono" style={{ textAlign: 'right' }}>
                  {identity.mention_count}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="tiny faint" style={{ marginTop: '0.7rem' }}>
          Contact details are never shown in the clear here. Whether a token is{' '}
          <em>present</em> is the fact worth showing; the address itself stays sealed.
        </p>
      </Card>

      <Card title="Recent activity" flush>
        {data.recent.map((item) => (
          <button className="list-item" key={item.doc_id} onClick={() => onOpen(item.doc_id)}>
            <SourceBadge provider={item.provider} />
            <div style={{ minWidth: 0, flex: 1 }}>
              <div className="truncate">{item.title || item.doc_id}</div>
              <div className="tiny faint">{relative(item.ts)}</div>
            </div>
          </button>
        ))}
        {!data.recent.length && <Empty title="No recent activity" />}
      </Card>
    </>
  )
}
