/** Shared presentational components. */

import type { CSSProperties, ReactNode } from 'react'
import { clsx, hueFor } from '../lib/format'

export function Card({
  title,
  hint,
  actions,
  children,
  flush,
  tight,
  style,
}: {
  title?: ReactNode
  hint?: ReactNode
  actions?: ReactNode
  children: ReactNode
  flush?: boolean
  tight?: boolean
  style?: CSSProperties
}) {
  return (
    <section className="card" style={style}>
      {title && (
        <header className="card-head">
          <h2>{title}</h2>
          {hint && <span className="hint">{hint}</span>}
          <div className="spacer" />
          {actions}
        </header>
      )}
      <div className={clsx('card-body', flush && 'flush', tight && 'tight')}>{children}</div>
    </section>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card">
      <div className="card-body">
        <div className="stat-label">{label}</div>
        <div className="stat-value">{value}</div>
        {sub && <div className="tiny faint">{sub}</div>}
      </div>
    </div>
  )
}

/** Source badge. Colour is keyed to the provider so a source is recognisable
 *  before its name is read, but the name is always present - the colour is a
 *  secondary cue, not the label. */
export function SourceBadge({ provider, label }: { provider: string; label?: string }) {
  return (
    <span
      className="badge source"
      style={{ background: `hsl(${hueFor(provider)} 58% 46%)` }}
      title={provider}
    >
      {label ?? provider}
    </span>
  )
}

export function Bar({ value, max = 1 }: { value: number; max?: number }) {
  const pct = Math.max(0, Math.min(100, (value / (max || 1)) * 100))
  return (
    <div className="bar" title={`${pct.toFixed(0)}%`}>
      <span style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Spinner() {
  return <span className="spinner" aria-label="loading" />
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <div className="tiny">{children}</div>}
    </div>
  )
}

export function ErrorNote({ error }: { error: { detail: string; status: number; requestId?: string } }) {
  return (
    <div className="evidence" style={{ borderColor: 'var(--danger)', background: 'var(--danger-soft)' }}>
      <strong className="tiny">{error.status || 'network'}</strong> {error.detail}
      {error.requestId && <div className="tiny faint">request {error.requestId}</div>}
    </div>
  )
}

export function Drawer({
  title,
  subtitle,
  onClose,
  children,
  actions,
}: {
  title: ReactNode
  subtitle?: ReactNode
  onClose: () => void
  children: ReactNode
  actions?: ReactNode
}) {
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true">
        <header className="drawer-head">
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontWeight: 620, letterSpacing: '-0.01em' }}>{title}</div>
            {subtitle && <div className="tiny faint">{subtitle}</div>}
          </div>
          {actions}
          <button className="btn sm" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        <div className="drawer-body">{children}</div>
      </aside>
    </>
  )
}
