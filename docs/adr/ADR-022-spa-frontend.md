# ADR-022: Vite SPA instead of Next.js

**Status:** accepted · **Area:** frontend · **Reversible:** yes, configuration

## The blueprint specifies

Next.js.

## We use

Vite + React + TypeScript, built to static files and served by the FastAPI
process at `/`.

## Why

**There is no server to render.** The client holds the keys. The server's job is
to seal, search and project; it has no session, no user record, and no
per-request markup worth having. Next.js's SSR exists to turn a request into
HTML for a crawler or a slow first paint, and a private workspace has neither.

**One process, one origin, no CORS.** FastAPI mounts `frontend/dist` and serves
`index.html` for client-side routes. The dev workflow still has CORS configured
for the Vite server on `:5173`, but a dev-only origin is a far smaller thing to
justify than a permissive production policy — and in the default deployment
there is no CORS at all.

**Faster to build and to iterate.** The production build is under a second, and
there is no server-component boundary to reason about when a view needs a
WebGL-ish D3 force simulation in it.

## What we gave up

**No SSR, so no server-rendered first paint.** The app is a JS bundle that
renders client-side. For an authenticated tool behind a login that is the right
trade; for a public marketing surface it would be the wrong one, and the ADR
would be reversed.

**No React Server Components.** The data flow is one-directional — fetch from
`/api`, render — which is simpler but means the server does no work to shape
the payload beyond what the routes return.

**Chunking is manual.** D3 is split into its own chunk (it is ~49 kB gzipped and
only the graph view needs it), via `manualChunks` in `vite.config.ts`. That is
the kind of thing a framework would have handled, and the cost of handling it
ourselves is one config entry.

## Bundle, for the record

```
dist/assets/index.js    58.6 kB │ gzip  17.5 kB   application
dist/assets/d3.js       48.7 kB │ gzip  16.8 kB   graph view only
dist/assets/react.js   140.8 kB │ gzip  45.2 kB   framework
dist/assets/index.css   11.9 kB │ gzip   3.3 kB
```
