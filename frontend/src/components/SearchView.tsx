/** Search view: keyword + natural language, with the plan shown next to the
 *  results.
 *
 *  The plan panel is the interesting part. A system that guesses what you meant
 *  and shows only the answer is asking for trust it has not earned; showing
 *  "I read this as: kind=file, after=2026-08-27, terms=[budget]" lets the user
 *  catch a misreading in one glance and fix it. Every NL search system should
 *  do this and most do not.
 */

import { useEffect, useState } from 'react'
import { api, type SearchResponse } from '../api'
import { useDebounced } from '../lib/hooks'
import { relative } from '../lib/format'
import { highlight } from '../lib/highlight'
import { Card, Empty, ErrorNote, SourceBadge, Spinner } from './ui'

const KINDS = ['message', 'email', 'file', 'note', 'video', 'transcript'] as const
const SOURCES = ['slack', 'gmail', 'discord', 'whatsapp', 'gdrive', 'notion', 'youtube'] as const

const EXAMPLES = [
  'what did we decide about the shard key',
  'budget file Alice sent',
  'anything else about retention policy',
  'what deadlines are coming up',
  'design review',
]

export function SearchView({ onOpen }: { onOpen: (docId: string) => void }) {
  const [text, setText] = useState('')
  const [kinds, setKinds] = useState<string[]>([])
  const [providers, setProviders] = useState<string[]>([])
  const [natural, setNatural] = useState(true)
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [plan, setPlan] = useState<{
    intent: string
    execution: string
    confidence: number
    explanation: string[]
    unsupported: string[]
  } | null>(null)
  const [error, setError] = useState<{ detail: string; status: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [took, setTook] = useState<number | undefined>()

  const debounced = useDebounced(text, 260)

  // One effect for both query styles, so the two paths can never disagree
  // about what the current filters are.
  useEffect(() => {
    const query = debounced.trim()
    const nothingTyped = !query && !kinds.length && !providers.length
    if (nothingTyped) {
      setResponse(null)
      setPlan(null)
      return
    }
    let cancelled = false
    setLoading(true)
    const run = natural
      ? api
          .nl2query(query || '(everything)', 20)
          .then((r) => {
            if (cancelled) return
            setPlan(r.plan)
            // The compiler is the authority on filters, so its plan replaces
            // the hand-rolled ones rather than being merged with them -
            // otherwise a UI filter and an inferred one both apply and the
            // result set is narrower than either intent.
            setResponse(r.result)
            setTook((r.result as (SearchResponse & { _tookMs?: number }) | null)?._tookMs)
          })
      : api
          .search({
            q: query,
            kinds: kinds.join(','),
            providers: providers.join(','),
            limit: 20,
          })
          .then((r) => {
            if (cancelled) return
            setPlan(null)
            setResponse(r)
            setTook(r._tookMs)
          })
    run
      .catch((err: { detail?: string; status?: number }) => {
        if (cancelled) return
        setError({ detail: err.detail ?? String(err), status: err.status ?? 0 })
        setResponse(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [debounced, kinds.join(','), providers.join(','), natural])

  function toggle(list: string[], value: string, set: (next: string[]) => void) {
    set(list.includes(value) ? list.filter((v) => v !== value) : [...list, value])
  }

  const hits = response?.hits ?? []
  const facets = response?.facets ?? {}
  const tookMs = response?.took_ms ?? took

  return (
    <>
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="card-body">
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <div className="searchbar">
              <input
                className="input"
                placeholder="Ask a question, or search across every source…"
                value={text}
                onChange={(event) => setText(event.target.value)}
                autoFocus
              />
            </div>
            <button
              className={natural ? 'btn primary' : 'btn'}
              onClick={() => setNatural((n) => !n)}
              title="Natural-language mode compiles a typed query plan and shows it below"
            >
              {natural ? 'Question mode' : 'Keyword mode'}
            </button>
          </div>

          <div className="row-wrap" style={{ marginBottom: '0.6rem' }}>
            <span className="tiny faint" style={{ minWidth: 52 }}>
              Kind
            </span>
            {KINDS.map((kind) => (
              <button
                key={kind}
                className="chip"
                aria-pressed={kinds.includes(kind)}
                onClick={() => toggle(kinds, kind, setKinds)}
              >
                {kind}
              </button>
            ))}
          </div>
          <div className="row-wrap">
            <span className="tiny faint" style={{ minWidth: 52 }}>
              Source
            </span>
            {SOURCES.map((source) => (
              <button
                key={source}
                className="chip"
                aria-pressed={providers.includes(source)}
                onClick={() => toggle(providers, source, setProviders)}
              >
                {source}
              </button>
            ))}
          </div>

          {!text && (
            <div className="row-wrap" style={{ marginTop: '0.75rem' }}>
              <span className="tiny faint">Try:</span>
              {EXAMPLES.map((example) => (
                <button key={example} className="chip" onClick={() => setText(example)}>
                  {example}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {error && <ErrorNote error={error} />}

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0,1fr) 296px' }}>
        <div className="col">
          {plan && (
            <Card title="Query plan" hint={`${plan.intent} · confidence ${plan.confidence.toFixed(2)}`}>
              <ul className="tiny muted" style={{ margin: 0, paddingLeft: '1.1rem' }}>
                {plan.explanation.map((line, index) => (
                  <li key={index}>{line}</li>
                ))}
              </ul>
              {plan.unsupported.length > 0 && (
                <div className="evidence" style={{ marginTop: '0.5rem' }}>
                  <div className="tiny" style={{ fontWeight: 600 }}>
                    Not expressible in the query language
                  </div>
                  {plan.unsupported.map((line, index) => (
                    <div className="tiny" key={index}>
                      {line}
                    </div>
                  ))}
                </div>
              )}
            </Card>
          )}

          <Card
            title={response ? `${response.total}${response.total_is_exact ? '' : '+'} results` : 'Results'}
            hint={
              tookMs !== undefined
                ? `${tookMs} ms · ${response?.engine ?? ''} · ${response?.index?.documents ?? 0} docs indexed`
                : undefined
            }
            actions={loading ? <Spinner /> : undefined}
            flush
          >
            {response?.degraded?.length ? (
              <div className="evidence" style={{ margin: '0.6rem 1rem' }}>
                {response.degraded.map((line, index) => (
                  <div className="tiny" key={index}>
                    {line}
                  </div>
                ))}
              </div>
            ) : null}
            {!response && !loading && (
              <Empty title="Search everything at once">
                Results are ranked by field-weighted BM25 fused with embeddings. Bodies are
                decrypted only for the results on screen.
              </Empty>
            )}
            {response && hits.length === 0 && (
              <Empty title="Nothing matched">
                The index has {response.index?.terms ?? 0} terms over{' '}
                {response.index?.documents ?? 0} documents. Try fewer words, or a source filter.
              </Empty>
            )}
            <div className="list">
              {hits.map((hit) => (
                <button className="list-item" key={hit.doc_id} onClick={() => onOpen(hit.doc_id)}>
                  <SourceBadge provider={hit.provider} />
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div className="row-wrap" style={{ marginBottom: '0.1rem' }}>
                      <span className="result-title truncate">
                        {highlight(
                          hit.title || hit.snippet || '(no title)',
                          hit.highlight_terms,
                        )}
                      </span>
                      {hit.has_attachment && <span className="badge">file</span>}
                      {hit.deadline_count > 0 && (
                        <span className="badge warn">{hit.deadline_count} deadline</span>
                      )}
                    </div>
                    <div className="result-snippet clamp-2">
                      {highlight(hit.snippet || '', hit.highlight_terms)}
                    </div>
                    <div className="tiny faint" style={{ marginTop: '0.2rem' }}>
                      {hit.sender_name ? `${hit.sender_name} · ` : ''}
                      {relative(hit.ts)}
                      {hit.lexical_rank !== null ? ` · lexical #${hit.lexical_rank}` : ''}
                      {hit.semantic_rank !== null ? ` · semantic #${hit.semantic_rank}` : ''}
                      {hit.person_ids.length > 0 ? ` · ${hit.person_ids.length} resolved people` : ''}
                    </div>
                  </div>
                  <div className="mono faint" style={{ minWidth: 44, textAlign: 'right' }}>
                    {hit.score.toFixed(3)}
                  </div>
                </button>
              ))}
            </div>
          </Card>
        </div>

        <div className="col">
          <Card title="Facets" hint="in the result set" flush>
            {!Object.keys(facets).length && (
              <div className="card-body tiny faint">No facets yet.</div>
            )}
            {Object.entries(facets).map(([name, buckets]) => (
              <div key={name} style={{ padding: '0.5rem 0.75rem', borderBottom: '1px solid var(--border)' }}>
                <div className="stat-label" style={{ marginBottom: '0.25rem' }}>
                  {name}
                </div>
                <div className="col" style={{ gap: '0.18rem' }}>
                  {buckets.slice(0, 8).map((bucket) => (
                    <div className="row tiny" key={bucket.value}>
                      <span className="truncate" style={{ flex: 1 }}>
                        {bucket.value}
                      </span>
                      <span className="mono faint">{bucket.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </Card>

          {response && (
            <Card title="How this was interpreted">
              <dl className="kv">
                <dt>Mode</dt>
                <dd>{response.parsed_query.natural_language ? 'question' : 'keyword'}</dd>
                <dt>Terms used</dt>
                <dd className="mono">
                  {(response.parsed_query.retrieval_terms ?? []).join(', ') || '—'}
                </dd>
                {response.parsed_query.stopwords_dropped?.length ? (
                  <>
                    <dt>Dropped</dt>
                    <dd className="mono faint">
                      {response.parsed_query.stopwords_dropped.join(', ')}
                    </dd>
                  </>
                ) : null}
                <dt>Index</dt>
                <dd className="mono">
                  {response.index?.documents}d / {response.index?.terms}t /{' '}
                  {response.index?.postings}p
                </dd>
              </dl>
              <p className="tiny faint" style={{ marginBottom: 0, marginTop: '0.6rem' }}>
                Function words are dropped from questions and kept for keyword queries — a
                keyword search for “to be” should find “to be”.
              </p>
            </Card>
          )}
        </div>
      </div>
    </>
  )
}
