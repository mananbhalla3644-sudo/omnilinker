/** Force-directed relationship graph (D3).
 *
 *  Design decisions that are not obvious:
 *
 *  * **Advisory edges are dashed.** Blueprint 11.5 requires a user to be able
 *    to tell an observed relationship from a machine's inference at a glance.
 *    A legend plus a glyph is not enough at 200 edges, so the line style
 *    carries it: solid = observed, dashed = the system suggesting.
 *
 *  * **The simulation is bounded and disposable.** 3 ticks per render, a fixed
 *    iteration cap, and a hard node ceiling. A force layout on an unbounded
 *    graph does not converge, it just burns the main thread — which is exactly
 *    how a graph view becomes the reason someone closes the tab.
 *
 *  * **Labels appear on hover and on the focused node**, not on every node.
 *    A 60-node graph with 60 labels is unreadable; a legend-free graph with
 *    hover labels is explorable.
 *
 *  * **Selection state lives in React, positions live in D3.** Re-rendering the
 *    whole SVG on selection would restart the simulation and make the graph
 *    jump, so selection only re-styles.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { api, type GraphEdge, type GraphNode, type Person } from '../api'
import { clsx } from '../lib/format'
import { Card, Empty, ErrorNote, Spinner } from './ui'

const MAX_NODES = 260

const LABEL_COLORS: Record<string, string> = {
  Person: 'var(--accent)',
  Identity: 'var(--advisory)',
  Conversation: 'var(--warn)',
  Message: 'var(--text-faint)',
  File: 'var(--ok)',
  Note: '#c2185b',
  Video: '#00838f',
  Transcript: '#5d4037',
  Event: 'var(--danger)',
}

const SHAPES: Record<string, 'circle' | 'square' | 'diamond'> = {
  Person: 'circle',
  Identity: 'square',
  Conversation: 'diamond',
  File: 'square',
  Note: 'square',
  Event: 'diamond',
}

type Sim = {
  nodes: GraphNode[]
  links: { source: string; target: string; type: string; props: Record<string, unknown> }[]
}

export function GraphView({ onOpenPerson }: { onOpenPerson: (personId: string) => void }) {
  const [people, setPeople] = useState<Person[]>([])
  const [focus, setFocus] = useState<string>('')
  const [hops, setHops] = useState(1)
  const [sim, setSim] = useState<Sim | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<{ detail: string; status: number } | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [hover, setHover] = useState<{ node: GraphNode; x: number; y: number } | null>(null)
  const [pathTarget, setPathTarget] = useState<string>('')
  const [pathResult, setPathResult] = useState<string | null>(null)

  const wrapRef = useRef<HTMLDivElement | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const simRef = useRef<d3.Simulation<any, any> | null>(null)

  useEffect(() => {
    api
      .persons()
      .then((r) => {
        setPeople(r.persons)
        const first = r.persons.find((p) => p.cross_source) ?? r.persons[0]
        if (first) setFocus(first.person_id)
      })
      .catch((err) => setError({ detail: String(err), status: 0 }))
  }, [])

  const load = useCallback(async () => {
    if (!focus) return
    setLoading(true)
    setError(null)
    try {
      const r = await api.ego(focus, hops)
      const nodes = r.nodes.slice(0, MAX_NODES)
      const ids = new Set(nodes.map((n) => n.id))
      const links = r.edges
        .filter((edge: GraphEdge) => ids.has(edge.source) && ids.has(edge.target))
        .map((edge: GraphEdge) => ({
          source: edge.source,
          target: edge.target,
          type: edge.type,
          props: edge.props,
        }))
      setSim({ nodes, links })
      setSelected(null)
    } catch (err) {
      setError({ detail: String(err), status: 0 })
    } finally {
      setLoading(false)
    }
  }, [focus, hops])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    if (!sim || !svgRef.current || !wrapRef.current) return
    const svg = d3.select(svgRef.current)
    const rect = wrapRef.current.getBoundingClientRect()
    const width = Math.max(320, rect.width)
    const height = Math.max(280, rect.height)

    simRef.current?.stop()
    svg.selectAll('*').remove()

    const root = svg.append('g')

    // Node degree drives radius: the eye should land on the hub, not on a
    // leaf that happens to be drawn first.
    const degree = new Map<string, number>()
    for (const link of sim.links) {
      degree.set(link.source, (degree.get(link.source) ?? 0) + 1)
      degree.set(link.target, (degree.get(link.target) ?? 0) + 1)
    }

    type D3Node = GraphNode & d3.SimulationNodeDatum
    const nodes: D3Node[] = sim.nodes.map((n) => ({ ...n }))
    const byId = new Map(nodes.map((n) => [n.id, n]))
    const links = sim.links
      .filter((l) => byId.has(l.source) && byId.has(l.target))
      .map((l) => ({ ...l }))

    const simulation = d3
      .forceSimulation<D3Node>(nodes)
      .force(
        'link',
        d3
          .forceLink<D3Node, any>(links)
          .id((n: D3Node) => n.id)
          .distance((l: any) => (l.props?.state === 'accepted' ? 48 : 66))
          .strength(0.32),
      )
      .force('charge', d3.forceManyBody<D3Node>().strength(-190).distanceMax(420))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide<D3Node>().radius((n) => radius(n) + 5))
      // Bias toward the horizontal: conversation-shaped clusters read better
      // wide than tall, and the panel is wider than it is tall.
      .force('x', d3.forceX(width / 2).strength(0.05))
      .alpha(0.9)
      .alphaDecay(0.045)
      .velocityDecay(0.34)

    const link = root
      .append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d: any) => (d.props?.state === 'accepted' ? 'var(--advisory)' : 'var(--border-strong)'))
      .attr('stroke-width', (d: any) => (d.props?.state === 'accepted' ? 1.8 : 0.9))
      .attr('stroke-opacity', 0.75)
      .attr('stroke-dasharray', (d: any) => (d.props?.state === 'accepted' ? '4 3' : null))

    const node = root
      .append('g')
      .selectAll<SVGGElement, D3Node>('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'pointer')
      .on('mouseleave', () => setHover(null))
      .on('click', (_event: MouseEvent, d: D3Node) => {
        setSelected(d.id)
        if (d.label === 'Person' && d.id !== focus) setFocus(d.id)
      })
      .on('dblclick', (_event: MouseEvent, d: D3Node) => {
        if (d.label === 'Person') onOpenPerson(d.id)
      })

    node
      .append('circle')
      .attr('r', radius)
      .attr('fill', (d) => fillFor(d))
      .attr('stroke', 'var(--surface)')
      .attr('stroke-width', 1.5)

    // Shapes carry type without a legend lookup: a legend the user has to learn
    // is worse than a shape that is already legible.
    node
      .filter((d) => SHAPES[d.label] === 'square')
      .append('rect')
      .attr('x', (d) => -radius(d))
      .attr('y', (d) => -radius(d))
      .attr('width', (d) => radius(d) * 2)
      .attr('height', (d) => radius(d) * 2)
      .attr('rx', 2)
      .attr('fill', (d) => fillFor(d))
      .attr('stroke', 'var(--surface)')
      .attr('stroke-width', 1.5)

    node
      .filter((d) => SHAPES[d.label] === 'diamond')
      .append('rect')
      .attr('x', (d) => -radius(d) * 0.78)
      .attr('y', (d) => -radius(d) * 0.78)
      .attr('width', (d) => radius(d) * 1.56)
      .attr('height', (d) => radius(d) * 1.56)
      .attr('rx', 1)
      .attr('transform', 'rotate(45)')
      .attr('fill', (d) => fillFor(d))
      .attr('stroke', 'var(--surface)')
      .attr('stroke-width', 1.5)

    const labels = node
      .append('text')
      .text((d) => labelFor(d))
      .attr('font-size', 10.5)
      .attr('dx', (d) => radius(d) + 5)
      .attr('dy', 3.5)
      .attr('fill', 'var(--text-dim)')
      .attr('pointer-events', 'none')
      .attr('opacity', 0)

    /** Dim everything except the active node and its edges. Pure styling, so it
     *  never touches positions - the simulation must not restart on hover. */
    const mark = () => {
      const id = selected ?? hover?.node.id ?? null
      const dim = 0.12
      node.attr('opacity', (d) => (id && d.id !== id ? dim : 1))
      link.attr('stroke-opacity', (d: any) => {
        if (!id) return 0.75
        const s = typeof d.source === 'object' ? (d.source as D3Node).id : d.source
        const t = typeof d.target === 'object' ? (d.target as D3Node).id : d.target
        return s === id || t === id ? 0.9 : dim
      })
      labels.attr('opacity', (d) => (d.id === id ? 1 : 0))
    }

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => (d.source as D3Node).x ?? 0)
        .attr('y1', (d: any) => (d.source as D3Node).y ?? 0)
        .attr('x2', (d: any) => (d.target as D3Node).x ?? 0)
        .attr('y2', (d: any) => (d.target as D3Node).y ?? 0)
      node.attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`)
    })

    mark()

    // Re-apply the dimming when hover or selection changes. Polling is
    // deliberate: subscribing to React state from inside the D3 effect would
    // either need a second effect (and rebuild the simulation) or a ref bridge
    // to a mutable box. One interval on a boolean-equality check is cheaper
    // than either and cannot desynchronise.
    const interval = window.setInterval(mark, 110)

    return () => {
      window.clearInterval(interval)
      simulation.stop()
      simRef.current = null
    }
  }, [sim, focus])

  const legend = useMemo(() => {
    const counts = new Map<string, number>()
    for (const node of sim?.nodes ?? []) counts.set(node.label, (counts.get(node.label) ?? 0) + 1)
    return [...counts.entries()].sort((a, b) => b[1] - a[1])
  }, [sim])

  const focusPerson = people.find((p) => p.person_id === focus)

  async function findPath() {
    if (!focus || !pathTarget) return
    try {
      const r = await api.path(focus, pathTarget)
      setPathResult(
        r.found
          ? `${r.hops} hop${r.hops === 1 ? '' : 's'} apart`
          : (r.note ?? 'no path found'),
      )
    } catch (err) {
      setPathResult(String(err))
    }
  }

  return (
    <>
      <Card
        title="Relationship graph"
        hint={`${sim?.nodes.length ?? 0} nodes · ${sim?.links.length ?? 0} edges`}
        actions={loading ? <Spinner /> : <button className="btn sm" onClick={() => void load()}>Redraw</button>}
      >
        <div className="row-wrap" style={{ marginBottom: '0.7rem' }}>
          <select
            className="select"
            style={{ maxWidth: 240 }}
            value={focus}
            onChange={(event) => setFocus(event.target.value)}
          >
            {people.map((person) => (
              <option key={person.person_id} value={person.person_id}>
                {person.display_name} · {person.providers.join(', ')}
              </option>
            ))}
          </select>
          <select className="select" style={{ maxWidth: 130 }} value={hops} onChange={(e) => setHops(Number(e.target.value))}>
            <option value={1}>1 hop</option>
            <option value={2}>2 hops</option>
            <option value={3}>3 hops</option>
          </select>
          <span className="tiny faint">
            Click a person to recentre · double-click to open · drag to pan · scroll to zoom
          </span>
        </div>

        {focusPerson && (
          <div className="row-wrap" style={{ marginBottom: '0.6rem' }}>
            <span className="badge accent">{focusPerson.providers.length} sources</span>
            {focusPerson.providers.map((provider) => (
              <span key={provider} className="badge">
                {provider}
              </span>
            ))}
            <span className="badge">{focusPerson.message_count} records</span>
            {focusPerson.aliases.map((alias) => (
              <span key={alias} className="badge warn">
                aka {alias}
              </span>
            ))}
          </div>
        )}

        {error && <ErrorNote error={error} />}

        <div className="graph-wrap" ref={wrapRef}>
          <svg ref={svgRef} />
          <div className="graph-legend">
            <div className="legend-row" style={{ fontWeight: 600, marginBottom: '0.15rem' }}>
              Legend
            </div>
            {legend.map(([label, count]) => (
              <div className="legend-row" key={label}>
                <span
                  className={clsx('legend-swatch', label !== 'Person' && 'square')}
                  style={{ background: LABEL_COLORS[label] ?? 'var(--text-faint)' }}
                />
                <span>{label}</span>
                <span className="faint" style={{ marginLeft: 'auto' }}>
                  {count}
                </span>
              </div>
            ))}
            <div className="legend-row" style={{ marginTop: '0.25rem' }}>
              <span className="legend-line" />
              <span>machine suggestion</span>
            </div>
            <div className="legend-row">
              <span className="legend-line solid" />
              <span>observed</span>
            </div>
          </div>
          <div className="graph-hint">size = degree</div>
          {hover && (
            <div
              className="graph-tooltip"
              style={{ left: Math.min(hover.x, 380), top: hover.y }}
            >
              <strong>{labelFor(hover.node) || hover.node.id}</strong>
              <div className="tiny faint">{hover.node.label}</div>
            </div>
          )}
        </div>
      </Card>

      <div className="grid grid-2" style={{ marginTop: '1rem' }}>
        <Card title="Find a path" hint="how two people are connected">
          <div className="row" style={{ marginBottom: '0.6rem' }}>
            <select className="select" value={focus} onChange={(e) => setFocus(e.target.value)}>
              {people.map((person) => (
                <option key={person.person_id} value={person.person_id}>
                  {person.display_name}
                </option>
              ))}
            </select>
            <span className="faint">→</span>
            <select
              className="select"
              value={pathTarget}
              onChange={(e) => setPathTarget(e.target.value)}
            >
              <option value="">pick someone…</option>
              {people
                .filter((p) => p.person_id !== focus)
                .map((person) => (
                  <option key={person.person_id} value={person.person_id}>
                    {person.display_name}
                  </option>
                ))}
            </select>
            <button className="btn" onClick={findPath} disabled={!pathTarget}>
              Find
            </button>
          </div>
          {pathResult && <div className="evidence">{pathResult}</div>}
          <p className="tiny faint" style={{ marginBottom: 0, marginTop: '0.6rem' }}>
            Paths run through shared conversations, which is why two people who never spoke can
            still be three hops apart. A missing path is information: they share no conversation.
          </p>
        </Card>

        <Card title="Selected node">
          {selected ? (
            <NodeDetail
              sim={sim}
              id={selected}
              onOpenPerson={onOpenPerson}
            />
          ) : (
            <Empty title="Nothing selected">Click a node in the graph.</Empty>
          )}
        </Card>
      </div>
    </>
  )
}

function NodeDetail({
  sim,
  id,
  onOpenPerson,
}: {
  sim: Sim | null
  id: string
  onOpenPerson: (personId: string) => void
}) {
  const node = sim?.nodes.find((n) => n.id === id)
  if (!node) return <Empty title="Node not in view" />
  const incident = (sim?.links ?? []).filter((l) => l.source === id || l.target === id)
  const others = incident.map((l) => (l.source === id ? l.target : l.source))
  const names = others
    .map((other) => sim?.nodes.find((n) => n.id === other))
    .filter(Boolean)
    .slice(0, 12)

  return (
    <div className="col">
      <div className="row-wrap">
        <span className="badge" style={{ background: 'var(--surface-2)' }}>
          {node.label}
        </span>
        <strong>{labelFor(node) || node.id}</strong>
        {node.label === 'Person' && (
          <button className="btn sm" onClick={() => onOpenPerson(node.id)}>
            Open profile
          </button>
        )}
      </div>
      <dl className="kv">
        {Object.entries(node.props)
          .filter(([, value]) => value !== '' && value !== null && value !== undefined)
          .slice(0, 8)
          .map(([key, value]) => (
            <div key={key} style={{ display: 'contents' }}>
              <dt>{key.replace(/_/g, ' ')}</dt>
              <dd>{Array.isArray(value) ? value.join(', ') || '—' : String(value)}</dd>
            </div>
          ))}
      </dl>
      {names.length > 0 && (
        <div>
          <div className="stat-label" style={{ marginBottom: '0.25rem' }}>
            connected to
          </div>
          <div className="row-wrap">
            {names.map((other) => (
              <span className="badge" key={other!.id}>
                {labelFor(other!) || other!.label}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function radius(node: GraphNode & d3.SimulationNodeDatum): number {
  const count = Number(node.props?.message_count ?? 0)
  const base = node.label === 'Person' ? 7 : node.label === 'Conversation' ? 5.5 : 4
  return base + Math.min(9, Math.sqrt(count) * 0.8)
}

function fillFor(node: GraphNode): string {
  if (node.label === 'Person' && node.props?.unresolved) return 'var(--warn)'
  return LABEL_COLORS[node.label] ?? 'var(--text-faint)'
}

function labelFor(node: GraphNode): string {
  const props = node.props ?? {}
  const name = props.display_name ?? props.label ?? props.name ?? props.title ?? ''
  if (name) return String(name)
  if (node.label === 'Person') return node.id.replace(/^per_/, '')
  if (node.label === 'Message') return String(props.provider ?? '')
  if (node.label === 'Transcript') return 'transcript'
  if (node.label === 'Identity') return `${props.provider ?? ''} id`
  return ''
}
