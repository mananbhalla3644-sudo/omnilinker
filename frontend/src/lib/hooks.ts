/** Small data-fetching primitives. No react-query: this app has seven endpoints
 * and a workspace-sized cache would be more code than the caching. */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../api'

export type Async<T> = {
  data: T | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

/**
 * Fetch on mount and whenever `deps` change.
 *
 * The `cancelled` flag matters more than it looks: without it, switching views
 * mid-flight sets state on an unmounted component, and in React 18 that logs a
 * warning *and* can apply a stale response from the previous view to the new
 * one. With a fast local API the race is easy to hit.
 */
export function useAsync<T>(
  loader: () => Promise<T>,
  deps: unknown[] = [],
  options: { immediate?: boolean } = {},
): Async<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(options.immediate !== false)
  const [nonce, setNonce] = useState(0)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  useEffect(() => {
    if (options.immediate === false) return
    let cancelled = false
    setLoading(true)
    loader()
      .then((value) => {
        if (cancelled || !alive.current) return
        setData(value)
        setError(null)
      })
      .catch((err: unknown) => {
        if (cancelled || !alive.current) return
        setError(err instanceof ApiError ? err : new ApiError(0, String(err)))
      })
      .finally(() => {
        if (!cancelled && alive.current) setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  return { data, error, loading, reload }
}

/** Debounce a rapidly changing value (search boxes, resize handlers). */
export function useDebounced<T>(value: T, ms = 220): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), ms)
    return () => window.clearTimeout(timer)
  }, [value, ms])
  return debounced
}

/** State that survives re-renders without triggering them. */
export function useRefState<T>(initial: T) {
  const ref = useRef(initial)
  return ref
}
