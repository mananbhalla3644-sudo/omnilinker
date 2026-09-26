/** Formatting helpers. All of them are locale-stable on purpose. */

const DATE_FMT = new Intl.DateTimeFormat(undefined, {
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})
const DAY_FMT = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' })
const FULL_FMT = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' })

export function parseTs(value: string | undefined | null): Date | null {
  if (!value) return null
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date
}

export function shortTime(value: string | undefined | null): string {
  const date = parseTs(value)
  return date ? DATE_FMT.format(date) : '—'
}

export function dayLabel(value: string | undefined | null): string {
  const date = parseTs(value)
  return date ? DAY_FMT.format(date) : '—'
}

export function fullTime(value: string | undefined | null): string {
  const date = parseTs(value)
  return date ? FULL_FMT.format(date) : '—'
}

export function relative(value: string | undefined | null, now = Date.now()): string {
  const date = parseTs(value)
  if (!date) return '—'
  const seconds = Math.round((date.getTime() - now) / 1000)
  const abs = Math.abs(seconds)
  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ['second', 60],
    ['minute', 3600],
    ['hour', 86400],
    ['day', 604800],
    ['week', 2629800],
    ['month', 31557600],
  ]
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })
  if (abs < 60) return rtf.format(seconds, 'second')
  let previous = 1
  for (const [unit, limit] of units) {
    if (abs < limit) return rtf.format(Math.round(seconds / previous), unit)
    previous = limit
  }
  return rtf.format(Math.round(seconds / 31557600), 'year')
}

export function bytes(value: number | undefined): string {
  const size = value ?? 0
  if (size < 1024) return `${size} B`
  const units = ['kB', 'MB', 'GB', 'TB']
  let scaled = size / 1024
  let index = 0
  while (scaled >= 1024 && index < units.length - 1) {
    scaled /= 1024
    index += 1
  }
  return `${scaled.toFixed(scaled < 10 ? 1 : 0)} ${units[index]}`
}

export function percent(value: number | undefined, digits = 0): string {
  return `${((value ?? 0) * 100).toFixed(digits)}%`
}

/** Deterministic hue per label, so a source keeps its colour across views. */
const HUES: Record<string, number> = {
  slack: 265,
  gmail: 5,
  discord: 235,
  whatsapp: 140,
  gdrive: 45,
  onedrive: 200,
  dropbox: 220,
  notion: 0,
  evernote: 30,
  youtube: 355,
}

export function hueFor(label: string | undefined): number {
  if (!label) return 0
  const known = HUES[label.toLowerCase()]
  if (known !== undefined) return known
  let hash = 0
  for (let i = 0; i < label.length; i += 1) hash = (hash * 31 + label.charCodeAt(i)) % 360
  return hash
}

export function clsx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}
