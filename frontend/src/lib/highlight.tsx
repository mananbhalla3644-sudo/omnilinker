import type { ReactNode } from 'react'

/** Highlight matches without HTML injection: split on term boundaries and wrap
 *  the odd segments in <mark>. Returns nodes rather than a string, so the
 *  caller cannot accidentally set innerHTML with user-controlled query text. */
export function highlight(text: string, terms: string[]): ReactNode[] {
  const cleaned = (terms ?? []).filter((t) => t && t.length > 2)
  if (!cleaned.length || !text) return [text]
  const pattern = cleaned
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .sort((a, b) => b.length - a.length)
    .join('|')
  const regex = new RegExp(`(${pattern})`, 'gi')
  return text.split(regex).map((part, index) =>
    index % 2 === 1 ? <mark key={index}>{part}</mark> : part,
  )
}
