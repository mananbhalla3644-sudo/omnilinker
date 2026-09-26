import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { useAsync } from './lib/hooks'
import { ConnectorsView } from './components/ConnectorsView'
import { DashboardView } from './components/DashboardView'
import { DocumentDrawer } from './components/DocumentDrawer'
import { FilesView } from './components/FilesView'
import { GraphView } from './components/GraphView'
import { PeopleView } from './components/PeopleView'
import { SearchView } from './components/SearchView'
import { TimelineView } from './components/TimelineView'
import { Spinner } from './components/ui'

type ViewId = 'dashboard' | 'search' | 'graph' | 'timeline' | 'people' | 'files' | 'connectors'

const NAV: { id: ViewId; label: string; icon: string; hint: string }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: '◧', hint: 'what needs attention' },
  { id: 'search', label: 'Search', icon: '⌕', hint: 'one index over every source' },
  { id: 'graph', label: 'Graph', icon: '◉', hint: 'who is connected to whom' },
  { id: 'timeline', label: 'Timeline', icon: '≡', hint: 'one clock, every source' },
  { id: 'people', label: 'People', icon: '☺', hint: 'resolved identities' },
  { id: 'files', label: 'Files', icon: '▤', hint: 'inventory and references' },
  { id: 'connectors', label: 'Connectors', icon: '⚙', hint: 'sources and passport' },
]

const TITLES: Record<ViewId, { title: string; sub: string }> = {
  dashboard: { title: 'Dashboard', sub: 'what needs attention, and what this system knows' },
  search: { title: 'Search', sub: 'field-weighted BM25 fused with embeddings' },
  graph: { title: 'Graph', sub: 'bounded traversal, observed vs. inferred edges' },
  timeline: { title: 'Timeline', sub: 'chronological across every connected source' },
  people: { title: 'People', sub: 'identity resolution with its evidence' },
  files: { title: 'Files', sub: 'inventory, owners, and reference counts' },
  connectors: { title: 'Connectors', sub: 'what will be read, and what happens to it' },
}

export default function App() {
  const [view, setView] = useState<ViewId>('dashboard')
  const [openDoc, setOpenDoc] = useState<string | null>(null)
  const [toast, setToast] = useState<string | null>(null)

  const health = useAsync(() => api.health(), [])
  const people = useAsync(() => api.persons(), [])

  // Deep link by hash. A private workspace has no SEO, but it does have
  // "send me the link to that message", and a view that cannot be linked is
  // annoying in exactly the situation the product exists for.
  useEffect(() => {
    const fromHash = window.location.hash.replace('#', '') as ViewId
    if (NAV.some((item) => item.id === fromHash)) setView(fromHash)
  }, [])

  const navigate = useCallback((next: string) => {
    const target = next as ViewId
    if (!NAV.some((item) => item.id === target)) return
    setView(target)
    window.location.hash = target
  }, [])

  useEffect(() => {
    const onHashChange = () => {
      const fromHash = window.location.hash.replace('#', '') as ViewId
      if (NAV.some((item) => item.id === fromHash)) setView(fromHash)
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const showToast = useCallback((message: string) => {
    setToast(message)
    window.setTimeout(() => setToast(null), 4200)
  }, [])

  const status = health.data?.status ?? 'unknown'
  const statusBadge =
    status === 'ok' ? (
      <span className="badge ok" title={health.data?.error || 'all derived stores healthy'}>
        healthy
      </span>
    ) : (
      <span className="badge danger" title={health.data?.error || 'a stage failed'}>
        degraded
      </span>
    )

  const isEmpty = (people.data?.persons.length ?? 0) === 0 && !health.loading

  return (
    <div className="shell">
      <nav className="sidebar">
        <div className="brand">
          <span className="brand-mark">OL</span>
          <span className="brand-name">OmniLinker</span>
          <span className="brand-version">0.1</span>
        </div>

        <div className="nav">
          <div className="nav-label">Workspace</div>
          {NAV.map((item) => (
            <button
              key={item.id}
              className="nav-item"
              aria-current={view === item.id ? 'page' : undefined}
              onClick={() => navigate(item.id)}
              title={item.hint}
            >
              <span className="nav-icon" aria-hidden>
                {item.icon}
              </span>
              {item.label}
              {item.id === 'people' && (people.data?.count ?? 0) > 0 && (
                <span className="nav-count">{people.data?.count}</span>
              )}
            </button>
          ))}
        </div>

        <div className="sidebar-foot">
          <div className="tiny faint" style={{ padding: '0 0.55rem', lineHeight: 1.5 }}>
            Content is sealed with AES-256-GCM. Nothing is ever written back to a provider.
          </div>
          {statusBadge}
        </div>
      </nav>

      <div className="main">
        <header className="topbar">
          <h1>{TITLES[view].title}</h1>
          <span className="topbar-sub">{TITLES[view].sub}</span>
          <div className="topbar-right">
            {health.loading ? <Spinner /> : statusBadge}
            <a className="btn sm" href="/api/docs" target="_blank" rel="noreferrer">
              API
            </a>
          </div>
        </header>

        <main className="content">
          {isEmpty && view !== 'connectors' && (
            <div className="card" style={{ marginBottom: '1rem' }}>
              <div className="card-body row">
                <div style={{ flex: 1 }}>
                  <strong>Nothing indexed yet.</strong>{' '}
                  <span className="muted">
                    Sync the synthetic demo workspace to populate search, the graph, identity
                    resolution and the detectors — no credentials needed.
                  </span>
                </div>
                <button className="btn primary" onClick={() => navigate('connectors')}>
                  Go to connectors
                </button>
              </div>
            </div>
          )}

          {view === 'dashboard' && (
            <DashboardView
              onOpen={setOpenDoc}
              onOpenPerson={() => navigate('people')}
              onNavigate={navigate}
            />
          )}
          {view === 'search' && <SearchView onOpen={setOpenDoc} />}
          {view === 'graph' && <GraphView onOpenPerson={() => navigate('people')} />}
          {view === 'timeline' && <TimelineView onOpen={setOpenDoc} />}
          {view === 'people' && <PeopleView onOpen={setOpenDoc} />}
          {view === 'files' && <FilesView onOpen={setOpenDoc} />}
          {view === 'connectors' && <ConnectorsView onToast={showToast} />}
        </main>
      </div>

      {openDoc && <DocumentDrawer docId={openDoc} onClose={() => setOpenDoc(null)} />}
      {toast && <div className="toast">{toast}</div>}
    </div>
  )
}
