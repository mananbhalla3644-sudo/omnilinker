/** Files: the inventory, with reference counts.
 *
 *  Reference count is the interesting column. A file with zero references is
 *  either an orphan (nobody discusses it) or invisible (people discuss it but
 *  we failed to match the attachment), and those are very different problems
 *  with very different fixes — so the count is always on screen.
 */

import { useState } from 'react'
import { api } from '../api'
import { useAsync, useDebounced } from '../lib/hooks'
import { bytes, relative } from '../lib/format'
import { Card, Empty, ErrorNote, SourceBadge, Spinner } from './ui'

const SOURCES = ['gdrive', 'onedrive', 'dropbox', 'slack', 'gmail', 'whatsapp'] as const

export function FilesView({ onOpen }: { onOpen: (docId: string) => void }) {
  const [providers, setProviders] = useState<string[]>([])
  const [text, setText] = useState('')
  const [onlyReferenced, setOnlyReferenced] = useState(false)
  const query = useDebounced(text, 250)
  const { data, error, loading } = useAsync(
    () => api.files({ providers: providers.join(',') || undefined, q: query || undefined }),
    [providers.join(','), query],
  )

  const files = (data?.files ?? []).filter((f) => !onlyReferenced || f.references > 0)
  const totalBytes = files.reduce((sum, f) => sum + (f.size_bytes ?? 0), 0)

  return (
    <Card
      title="Files"
      hint={`${files.length} of ${data?.total ?? 0} · ${bytes(totalBytes)}`}
      actions={loading ? <Spinner /> : undefined}
      flush
    >
      <div className="card-body" style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="row-wrap">
          <input
            className="input"
            style={{ maxWidth: 260 }}
            placeholder="Filter by name…"
            value={text}
            onChange={(event) => setText(event.target.value)}
          />
          {SOURCES.map((source) => (
            <button
              key={source}
              className="chip"
              aria-pressed={providers.includes(source)}
              onClick={() =>
                setProviders((prev) =>
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
            aria-pressed={onlyReferenced}
            onClick={() => setOnlyReferenced((v) => !v)}
            title="A file with no references is either an orphan or invisible — different problems, different fixes"
          >
            discussed only
          </button>
        </div>
      </div>

      {error && <ErrorNote error={error} />}
      {!loading && !files.length && (
        <Empty title="No files match">Clear the filters, or sync a drive connector.</Empty>
      )}

      <table className="plain">
        <thead>
          <tr>
            <th>Name</th>
            <th style={{ width: 110 }}>Source</th>
            <th style={{ width: 74 }}>Size</th>
            <th style={{ width: 110 }}>Referenced</th>
            <th style={{ width: 110 }}>Modified</th>
          </tr>
        </thead>
        <tbody>
          {files.map((file) => (
            <tr key={file.doc_id} style={{ cursor: 'pointer' }} onClick={() => onOpen(file.doc_id)}>
              <td>
                <div className="truncate" style={{ fontWeight: 540 }}>
                  {file.name}
                </div>
                <div className="tiny faint truncate">
                  {file.mime}
                  {file.folder_path?.length ? ` · ${file.folder_path.join(' / ')}` : ''}
                </div>
              </td>
              <td>
                <SourceBadge provider={file.provider} />
              </td>
              <td className="mono">{bytes(file.size_bytes)}</td>
              <td>
                {file.references > 0 ? (
                  <span className="badge ok">{file.references}×</span>
                ) : (
                  <span className="badge warn">orphan</span>
                )}
              </td>
              <td className="tiny faint">{relative(file.modified_ts)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}
