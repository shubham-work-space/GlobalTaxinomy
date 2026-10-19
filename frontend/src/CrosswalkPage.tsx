import { GitCompareArrows, Loader2, Search } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import type {
  CrosswalkDirection,
  CrosswalkMatch,
  CrosswalkResult,
  OccupationSuggestion,
  RankingSource,
} from './types'

function sourceDataset(direction: CrosswalkDirection): 'onet' | 'esco' {
  return direction === 'onet_to_esco' ? 'onet' : 'esco'
}

function pct(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '—'
  return `${(n * 100).toFixed(1)}%`
}

function rankingLabel(source: RankingSource | undefined, cacheOrigin?: string | null): string {
  if (source === 'embedding') return 'Embedding Top-10 (Top-1 ≥ threshold)'
  if (source === 'jev') return 'Jev ranked Top-10'
  if (source === 'jev_cached') {
    if (cacheOrigin?.startsWith('jev_cached_run:')) {
      return `Cached Jev Top-10 (${cacheOrigin.replace('jev_cached_run:', '')})`
    }
    if (cacheOrigin === 'jev_cached_live') return 'Cached Jev Top-10 (prior live lookup)'
    return 'Cached Jev Top-10'
  }
  return 'Top-10'
}

export default function CrosswalkPage() {
  const [direction, setDirection] = useState<CrosswalkDirection>('onet_to_esco')
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<OccupationSuggestion[]>([])
  const [selected, setSelected] = useState<OccupationSuggestion | null>(null)
  const [loadingSearch, setLoadingSearch] = useState(false)
  const [loadingMatch, setLoadingMatch] = useState(false)
  const [result, setResult] = useState<CrosswalkResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (query.trim().length < 2) {
      setSuggestions([])
      return
    }
    const handle = window.setTimeout(async () => {
      setLoadingSearch(true)
      try {
        const ds = sourceDataset(direction)
        const res = await fetch(
          `/api/occupations/${ds}?q=${encodeURIComponent(query.trim())}&limit=12`,
        )
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        setSuggestions((data.results || []) as OccupationSuggestion[])
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Search failed')
      } finally {
        setLoadingSearch(false)
      }
    }, 250)
    return () => window.clearTimeout(handle)
  }, [query, direction])

  const runMatch = useCallback(
    async (occupation: OccupationSuggestion) => {
      setSelected(occupation)
      setLoadingMatch(true)
      setError(null)
      setResult(null)
      try {
        const res = await fetch('/api/crosswalk', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            direction,
            occupation_id: occupation.occupation_id,
            top_k: 10,
            threshold: 0.6,
            model: 'P',
          }),
        })
        const data = await res.json().catch(() => ({}))
        if (!res.ok) {
          throw new Error(data?.detail?.message || data?.message || `HTTP ${res.status}`)
        }
        setResult(data as CrosswalkResult)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Match failed')
      } finally {
        setLoadingMatch(false)
      }
    },
    [direction],
  )

  const matches: CrosswalkMatch[] =
    result?.matches ?? result?.ranked?.matches ?? result?.embedding?.matches?.slice(0, 10) ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-2xl font-bold text-[var(--color-foam)]">
          Live crosswalk
        </h1>
        <p className="mt-1 text-sm text-[var(--color-mist)]">
          Model P Top-20 → Top-10. If Top-1 &lt; 0.60, Jev ranks candidates (reuses prior
          results when available).
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {(
          [
            ['onet_to_esco', 'O*NET → ESCO'],
            ['esco_to_onet', 'ESCO → O*NET'],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => {
              setDirection(id)
              setSelected(null)
              setResult(null)
              setSuggestions([])
            }}
            className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-wider ${
              direction === id
                ? 'bg-[var(--color-signal)] text-[var(--color-ink)]'
                : 'border border-[var(--color-line)] text-[var(--color-mist)]'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="rounded-2xl border border-[var(--color-line)] bg-[var(--color-panel)]/90 p-5">
        <label className="mb-2 block text-xs uppercase tracking-wider text-[var(--color-mist)]">
          Search occupation
        </label>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--color-mist)]" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type a title…"
            className="w-full rounded-xl border border-[var(--color-line)] bg-[var(--color-panel-2)] py-3 pl-10 pr-4 text-sm text-[var(--color-foam)] outline-none focus:border-[var(--color-signal)]"
          />
          {loadingSearch ? (
            <Loader2 className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 animate-spin text-[var(--color-mist)]" />
          ) : null}
        </div>
        {suggestions.length > 0 ? (
          <ul className="mt-3 max-h-56 overflow-y-auto rounded-xl border border-[var(--color-line)]">
            {suggestions.map((s) => (
              <li key={s.occupation_id}>
                <button
                  type="button"
                  onClick={() => void runMatch(s)}
                  className="flex w-full items-start gap-2 px-4 py-3 text-left text-sm hover:bg-[var(--color-panel-2)]"
                >
                  <GitCompareArrows className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-signal)]" />
                  <span>
                    <span className="block text-[var(--color-foam)]">
                      {s.preferred_title || s.occupation_id}
                    </span>
                    <span className="block text-xs text-[var(--color-mist)]">
                      {s.occupation_id}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </div>

      {error ? (
        <p className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </p>
      ) : null}

      {loadingMatch ? (
        <div className="flex items-center gap-2 text-sm text-[var(--color-mist)]">
          <Loader2 className="h-4 w-4 animate-spin" /> Matching (may call Jev if Top-1 &lt; 0.60)…
        </div>
      ) : null}

      {selected && matches.length > 0 ? (
        <div className="rounded-2xl border border-[var(--color-line)] bg-[var(--color-panel)]/90 p-5">
          <p className="text-xs uppercase tracking-wider text-[var(--color-mist)]">
            Source
          </p>
          <p className="mt-1 font-medium text-[var(--color-foam)]">
            {selected.preferred_title || selected.occupation_id}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-[var(--color-mist)]">
            <span>
              Top-1 similarity:{' '}
              <span className="font-semibold text-[var(--color-foam)]">
                {pct(result?.top1_similarity)}
              </span>
            </span>
            <span>
              Threshold:{' '}
              <span className="font-semibold text-[var(--color-foam)]">
                {pct(result?.threshold ?? 0.6)}
              </span>
            </span>
            <span className="rounded-full border border-[var(--color-line)] px-2 py-0.5 text-[var(--color-signal)]">
              {rankingLabel(result?.ranking_source, result?.jev?.cache_origin)}
            </span>
          </div>

          <p className="mt-5 text-xs uppercase tracking-wider text-[var(--color-mist)]">
            Top 10 matches
          </p>
          <ol className="mt-2 space-y-2">
            {matches.map((m, idx) => (
              <li
                key={`${m.occupation_id}-${idx}`}
                className={`rounded-xl border px-4 py-3 ${
                  idx === 0
                    ? 'border-[var(--color-signal)]/40 bg-[var(--color-panel-2)]'
                    : 'border-[var(--color-line)] bg-[var(--color-panel-2)]/60'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-[10px] uppercase tracking-wider text-[var(--color-mist)]">
                      #{idx + 1}
                    </p>
                    <p className="font-semibold text-[var(--color-foam)]">
                      {m.preferred_title || m.occupation_id}
                    </p>
                    <p className="mt-1 truncate text-xs text-[var(--color-mist)]">
                      {m.occupation_id}
                    </p>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className="text-base font-bold text-[var(--color-signal)]">
                      {pct(m.weighted_score)}
                    </p>
                    {m.ground_truth_match_type === 'exactMatch' ? (
                      <span className="mt-1 inline-block rounded-full border border-emerald-400/40 bg-emerald-400/15 px-2 py-0.5 text-[10px] uppercase tracking-wider text-emerald-300">
                        Exact
                      </span>
                    ) : m.ground_truth_match_type ? (
                      <span className="mt-1 inline-block rounded-full border border-[var(--color-line)] px-2 py-0.5 text-[10px] uppercase tracking-wider text-[var(--color-mist)]">
                        {m.ground_truth_match_type}
                      </span>
                    ) : null}
                  </div>
                </div>
              </li>
            ))}
          </ol>
          {result?.jev?.jev_error ? (
            <p className="mt-3 text-xs text-amber-200/90">
              Jev note: {result.jev.jev_error} (showing fallback order)
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
