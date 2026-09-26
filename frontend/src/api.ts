/**
 * API client.
 *
 * One function, one place that knows the wire format. Two rules it enforces:
 *
 * 1. **Errors are typed, not strings.** A failed request throws an
 *    `ApiError` carrying the status and the server's message, so a view can
 *    distinguish "no results" from "the backend is down" - a distinction the
 *    user absolutely needs and a bare `catch` always loses.
 *
 * 2. **Server latency is surfaced.** Every response carries `X-Omni-Took-Ms`,
 *    which is attached to the returned object. The UI displays it, because a
 *    product with a p95 budget should show the user what they are spending.
 */

export class ApiError extends Error {
  readonly status: number
  readonly detail: string
  readonly requestId?: string

  constructor(status: number, detail: string, requestId?: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.requestId = requestId
  }
}

export type Timed<T> = T & { _tookMs?: number }

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

async function request<T>(path: string, init?: RequestInit): Promise<Timed<T>> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  const took = Number(response.headers.get('X-Omni-Took-Ms') ?? '0') || undefined
  const text = await response.text()
  let body: unknown = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    throw new ApiError(response.status, text.slice(0, 300) || response.statusText)
  }
  if (!response.ok) {
    const payload = body as { detail?: string; error?: { message?: string; request_id?: string } }
    throw new ApiError(
      response.status,
      payload?.detail ?? payload?.error?.message ?? response.statusText,
      payload?.error?.request_id,
    )
  }
  return { ...(body as T), _tookMs: took } as Timed<T>
}

const get = <T,>(path: string, params: Record<string, string | number | boolean | undefined> = {}) =>
  request<T>(path + qs(params))

const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })

// --- types -----------------------------------------------------------------

export type SearchHit = {
  doc_id: string
  score: number
  rrf_score: number
  kind: string
  provider: string
  ts: string
  title: string
  sender_name: string
  snippet: string
  highlight_terms: string[]
  person_ids: string[]
  conversation_id: string
  lexical_rank: number | null
  semantic_rank: number | null
  has_attachment: boolean
  deadline_count: number
}

export type SearchResponse = {
  hits: SearchHit[]
  total: number
  total_is_exact: boolean
  took_ms: number
  facets: Record<string, { value: string; count: number }[]>
  parsed_query: {
    text: string
    terms: string[]
    retrieval_terms: string[]
    phrases: string[]
    exclude: string[]
    filters: Record<string, string[]>
    natural_language: boolean
    stopwords_dropped: string[]
  }
  engine: string
  index: Record<string, number>
  degraded: string[]
}

export type GraphNode = {
  id: string
  label: string
  props: Record<string, unknown>
}

export type GraphEdge = {
  source: string
  target: string
  type: string
  props: Record<string, unknown>
  key: string[]
}

export type Person = {
  person_id: string
  display_name: string
  aliases: string[]
  providers: string[]
  cross_source: boolean
  message_count: number
  identity_count: number
  contact_count: number
  last_seen: string
}

export type Document = Record<string, unknown> & {
  _id: string
  provider: string
  kind?: string
  sender_name?: string
  body_text?: string
  title?: string
  name?: string
  ts?: string
  extra?: Record<string, unknown>
  entities?: Record<string, unknown>
  attachments?: { name?: string; mime?: string }[]
  lineage?: Record<string, unknown>
}

export type Insight = {
  _id: string
  kind: string
  detector: string
  title: string
  detail: string
  entities: string[]
  confidence: number
  evidence: Record<string, unknown>[]
  state: string
}

export type Prediction = {
  _id: string
  kind: string
  title: string
  detail: string
  due: string
  confidence: number
  score: number
  source_ids: string[]
  evidence: Record<string, unknown>[]
  state: string
}

export type SystemInfo = {
  mode: string
  workspace: string
  stores: {
    backend: string
    documents: Record<string, number>
    graph: Record<string, unknown>
  }
  capabilities: Record<string, unknown>
  search: Record<string, number>
  deviations: {
    id: string
    area: string
    blueprint: string
    actual: string
    reason: string
    risk: string
  }[]
}

export type Connector = {
  id: string
  display_name: string
  auth_flow: string
  /** Null when the provider uses no consent screen (API key or file import). */
  authorize_url: string | null
  token_url: string | null
  scopes: { name: string; justification: string; sensitivity: string }[]
  content_kinds: string[]
  realtime_webhooks: boolean
  incremental_cursor: boolean
  supports_tombstones: boolean
  rate_limit_per_sec: number
  notes: string
  docs_url: string
}

export type FileItem = {
  doc_id: string
  name: string
  mime: string
  size_bytes: number
  provider: string
  modified_ts: string
  folder_path: string[]
  owner_person_id: string
  references: number
}

export type TimelineEvent = {
  doc_id: string
  ts: string
  provider: string
  kind: string
  title: string
  sender_name: string
  importance: number
  has_attachment: boolean
  deadline_count: number
  person_ids: string[]
}

export type QueryPlan = {
  plan_id: string
  intent: string
  execution: string
  compiled_by: string
  confidence: number
  request: Record<string, unknown>
  filters: unknown[]
  explanation: string[]
  unsupported: string[]
  person_hint: string
  subject: string
  signals: string[]
  alternatives: string[]
}

export type Suggestion = {
  _id: string
  identity_a: string
  identity_b: string
  display_name_a: string
  display_name_b: string
  provider_a: string
  provider_b: string
  score: number
  evidence: { feature: string; weight: number; detail: string }[]
  conflicts: { feature: string; weight: number; detail: string }[]
  status: string
}

// --- endpoints -------------------------------------------------------------

export const api = {
  systemInfo: () => get<SystemInfo>('/system/info'),
  health: () => get<{ status: string; stage: string; last_run_at: string; error: string }>(
    '/system/health',
  ),
  passport: () => get<{ passport: Record<string, unknown> }>('/passport'),

  connectors: () => get<{ connectors: Connector[] }>('/connectors'),
  sync: (connectorId = 'demo') =>
    post<{
      ingest: { status: string; counters: Record<string, number>; error_count: number }
      derived: Record<string, Record<string, unknown>>
      took_ms: number
    }>('/sync', { connector_id: connectorId }),

  search: (params: {
    q: string
    kinds?: string
    providers?: string
    person_id?: string
    after?: string
    before?: string
    has_attachment?: boolean
    limit?: number
    offset?: number
  }) => get<SearchResponse>('/search', params),

  suggest: (prefix: string) =>
    get<{ suggestions: { kind: string; value: string; count: number }[] }>(
      '/search/suggest',
      { prefix },
    ),

  nl2query: (question: string, limit = 20) =>
    post<{ plan: QueryPlan; result: SearchResponse | null; dispatch?: string }>('/nl2query', {
      question,
      limit,
    }),

  ego: (personId: string, hops = 1, limit = 140) =>
    get<{ nodes: GraphNode[]; edges: GraphEdge[]; node_count: number; edge_count: number }>(
      '/graph/ego',
      { person_id: personId, depth: hops, limit },
    ),

  path: (sourceId: string, targetId: string) =>
    get<{ found: boolean; hops: number | null; paths: unknown[]; note?: string }>('/graph/path', {
      source_id: sourceId,
      target_id: targetId,
      max_depth: 4,
    }),

  persons: (crossSourceOnly = false) =>
    get<{ persons: Person[]; count: number }>('/persons', { cross_source_only: crossSourceOnly }),
  person: (id: string) =>
    get<{
      person: Person
      identities: {
        identity_id: string
        provider: string
        display_name: string
        email_present: boolean
        phone_present: boolean
        mention_count: number
      }[]
      pending_suggestions: Suggestion[]
      recent: { doc_id: string; ts: string; title: string; provider: string }[]
    }>(`/persons/${encodeURIComponent(id)}`),
  suggestions: () => get<{ suggestions: Suggestion[]; count: number }>('/suggestions'),

  insights: (state = 'advisory') => get<{ insights: Insight[]; count: number }>('/insights', { state }),
  acceptInsight: (id: string, sourceId: string, targetId: string, kind: string) =>
    post(`/insights/${encodeURIComponent(id)}/accept`, {
      source_id: sourceId,
      target_id: targetId,
      kind,
    }),

  predictions: () => get<{ predictions: Prediction[]; count: number }>('/predictions'),
  files: (params: { providers?: string; q?: string; limit?: number; offset?: number } = {}) =>
    get<{ files: FileItem[]; total: number; has_more: boolean }>('/files', params),
  timeline: (params: { limit?: number; providers?: string; person_id?: string } = {}) =>
    get<{ events: TimelineEvent[]; count: number }>('/timeline', params),

  document: (id: string) =>
    get<{ document: Document; related: { doc_id: string; ts: string; provider: string; title: string }[] }>(
      `/documents/${id}`,
    ),
  summarize: (id: string) =>
    post<{
      text: string
      method: string
      compression: number
      source_words: number
      summary_words: number
      sentences: { text: string; score: number; position: number }[]
    }>(`/documents/${id}/summarize`, { max_sentences: 3 }),

  dropDerived: () => post<Record<string, unknown>>('/admin/drop-derived?rebuild=true'),
}
