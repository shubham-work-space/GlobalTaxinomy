import { Loader2, Play, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import RunConfigPanel from './RunConfigPanel'
import RunProgress from './RunProgress'
import type { RunConfig, RunProgressState, RunSummary } from './types'

function pct(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '—'
  return `${(n * 100).toFixed(2)}%`
}

function fmtWeights(weights: Record<string, number> | undefined): string {
  if (!weights) return '—'
  const abbrev: Record<string, string> = {
    label_max: 'Tmx',
    label_median: 'Tmd',
    description_pool: 'D',
    skills_pool: 'S',
    knowledge_pool: 'K',
  }
  return Object.entries(weights)
    .map(([k, v]) => `${abbrev[k] || k}${Math.round(v * 100)}`)
    .join(' ')
}

export default function EvaluationPage() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<RunProgressState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<RunSummary | null>(null)

  const loadRuns = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/runs')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = (await res.json()) as { runs: RunSummary[] }
      setRuns(data.runs || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load runs')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadRuns()
  }, [loadRuns])

  useEffect(() => {
    if (!busy && progress?.status !== 'running') return
    const es = new EventSource('/api/runs/progress')
    es.onmessage = (ev) => {
      try {
        const next = JSON.parse(ev.data) as RunProgressState
        setProgress(next)
        if (next.status === 'completed' || next.status === 'failed') {
          setBusy(false)
          void loadRuns()
          es.close()
        }
      } catch {
        /* ignore */
      }
    }
    es.onerror = () => {
      es.close()
    }
    return () => es.close()
  }, [busy, progress?.status, loadRuns])

  const handleStart = async (config: RunConfig) => {
    setBusy(true)
    setError(null)
    setProgress({
      status: 'running',
      run_id: null,
      label: null,
      phase: 'generate_embeddings',
      percent: 0,
      message: 'Starting…',
    })
    const res = await fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok) {
      setBusy(false)
      throw new Error(data?.detail?.message || data?.message || `HTTP ${res.status}`)
    }
    if (data.reused) {
      setBusy(false)
      setProgress({
        status: 'completed',
        run_id: data.run?.run_id ?? null,
        label: data.run?.label ?? null,
        phase: 'save_run',
        percent: 100,
        message: data.run?.message || 'Reused cached results',
      })
      await loadRuns()
      return
    }
    setProgress((prev) => ({
      ...(prev || {
        status: 'running',
        run_id: null,
        label: null,
        phase: null,
        percent: 0,
        message: '',
      }),
      run_id: data.run?.run_id ?? null,
      label: data.run?.label ?? null,
      status: 'running',
    }))
  }

  const openRun = async (run: RunSummary) => {
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(run.run_id)}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setSelected(data.run as RunSummary)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load run')
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="font-display text-2xl font-bold text-[var(--color-foam)]">
            Evals dashboard
          </h1>
          <p className="mt-1 text-sm text-[var(--color-mist)]">
            Model P + Jev/Ejento hybrid · Accuracy@1 (and @2–@10 when ranked)
          </p>
        </div>
        <button
          type="button"
          onClick={() => void loadRuns()}
          className="inline-flex items-center gap-2 rounded-full border border-[var(--color-line)] px-4 py-2 text-xs font-semibold uppercase tracking-wider text-[var(--color-foam)]"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>

      <RunConfigPanel onStart={handleStart} busy={busy} />
      <RunProgress progress={progress} />

      {error ? (
        <p className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </p>
      ) : null}

      <div className="overflow-x-auto rounded-2xl border border-[var(--color-line)] bg-[var(--color-panel)]/90">
        {loading ? (
          <div className="flex items-center gap-2 px-5 py-8 text-sm text-[var(--color-mist)]">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading runs…
          </div>
        ) : runs.length === 0 ? (
          <div className="px-5 py-8 text-sm text-[var(--color-mist)]">
            No runs yet. Configure weights/threshold above and start a run.
          </div>
        ) : (
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-[var(--color-line)] text-xs uppercase tracking-wider text-[var(--color-mist)]">
              <tr>
                <th className="px-4 py-3">Run</th>
                <th className="px-4 py-3">Resolver</th>
                <th className="px-4 py-3">N</th>
                <th className="px-4 py-3">Acc@1</th>
                <th className="px-4 py-3">Emb ≥thr</th>
                <th className="px-4 py-3">Resolver subset</th>
                <th className="px-4 py-3">Acc@2</th>
                <th className="px-4 py-3">Acc@3</th>
                <th className="px-4 py-3">Acc@5</th>
                <th className="px-4 py-3">Acc@10</th>
                <th className="px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const m = run.metrics_summary || {}
                const acc1 =
                  m.accuracy_at_1 ??
                  m.exact_match_at_1 ??
                  m.final_accuracy_after_llm ??
                  null
                const resolver = (m.resolver || 'jev').toString()
                const hiN = m.high_confidence_records
                const hiOk = m.high_confidence_correct
                const subN =
                  m.resolver_subset_records ??
                  m.records_sent_to_llm ??
                  m.records_sent_to_jev
                const subOk = m.resolver_subset_correct
                const subAcc =
                  m.resolver_subset_accuracy ??
                  m.llm_subset_accuracy_after ??
                  m.jev_subset_accuracy_after ??
                  null
                return (
                  <tr
                    key={run.run_id}
                    className="border-b border-[var(--color-line)]/60 text-[var(--color-foam)]"
                  >
                    <td className="px-4 py-3">
                      <div className="text-sm font-medium">{run.label}</div>
                      <div className="mt-1 font-mono text-xs text-[var(--color-mist)]">
                        {fmtWeights(run.weights)}
                      </div>
                      <div className="mt-1 text-xs text-[var(--color-mist)]">
                        thr {run.threshold} · {run.status}
                      </div>
                    </td>
                    <td className="px-4 py-3 capitalize">{resolver}</td>
                    <td className="px-4 py-3">{m.total_records ?? '—'}</td>
                    <td className="px-4 py-3 font-semibold">{pct(acc1)}</td>
                    <td className="px-4 py-3 text-xs">
                      {hiOk != null && hiN != null ? (
                        <>
                          <div>
                            {hiOk}/{hiN}
                          </div>
                          <div className="text-[var(--color-mist)]">
                            {pct(m.high_confidence_accuracy)}
                          </div>
                        </>
                      ) : (
                        pct(m.embedding_top1_accuracy)
                      )}
                    </td>
                    <td className="px-4 py-3 text-xs">
                      {subOk != null && subN != null ? (
                        <>
                          <div>
                            {subOk}/{subN}
                          </div>
                          <div className="text-[var(--color-mist)]">{pct(subAcc)}</div>
                        </>
                      ) : (
                        pct(subAcc)
                      )}
                    </td>
                    <td className="px-4 py-3">{pct(m.accuracy_at_2)}</td>
                    <td className="px-4 py-3">{pct(m.accuracy_at_3)}</td>
                    <td className="px-4 py-3">{pct(m.accuracy_at_5)}</td>
                    <td className="px-4 py-3">{pct(m.accuracy_at_10)}</td>
                    <td className="px-4 py-3">
                      <button
                        type="button"
                        onClick={() => void openRun(run)}
                        className="inline-flex items-center gap-1 rounded-full border border-[var(--color-line)] px-3 py-1 text-xs text-[var(--color-foam)]"
                      >
                        <Play className="h-3 w-3" />
                        Detail
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {selected ? (
        <div className="rounded-2xl border border-[var(--color-line)] bg-[var(--color-panel)]/90 px-5 py-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-display text-lg font-semibold text-[var(--color-foam)]">
              {selected.label} detail
            </h2>
            <button
              type="button"
              onClick={() => setSelected(null)}
              className="text-xs text-[var(--color-mist)]"
            >
              Close
            </button>
          </div>
          <pre className="overflow-x-auto rounded-xl bg-[var(--color-panel-2)] p-4 text-xs text-[var(--color-foam)]">
            {JSON.stringify(
              {
                run_id: selected.run_id,
                weights: selected.weights,
                threshold: selected.threshold,
                metrics: selected.metrics || selected.metrics_summary,
                message: selected.message,
              },
              null,
              2,
            )}
          </pre>
        </div>
      ) : null}
    </div>
  )
}
