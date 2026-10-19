import { Loader2, Play } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { RunConfig, TaxonomiesResponse } from './types'

const SIGNAL_LABELS: Record<string, string> = {
  label_max: 'Tmx (Title max)',
  label_median: 'Tmd (Title median)',
  description_pool: 'D (Description)',
  skills_pool: 'S (Skills)',
  knowledge_pool: 'K (Knowledge)',
}

const SUM_TOLERANCE = 0.001

type Props = {
  onStart: (config: RunConfig) => Promise<void>
  busy?: boolean
}

export default function RunConfigPanel({ onStart, busy = false }: Props) {
  const [tax, setTax] = useState<TaxonomiesResponse | null>(null)
  const [inputTaxonomy, setInputTaxonomy] = useState('onet')
  const [outputTaxonomy, setOutputTaxonomy] = useState('esco')
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [threshold, setThreshold] = useState(0.6)
  const [force, setForce] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void (async () => {
      try {
        const res = await fetch('/api/taxonomies')
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = (await res.json()) as TaxonomiesResponse
        setTax(data)
        setInputTaxonomy(data.default_input_taxonomy)
        setOutputTaxonomy(data.default_output_taxonomy)
        const pair =
          data.pairs.find(
            (p) =>
              p.input_taxonomy === data.default_input_taxonomy &&
              p.output_taxonomy === data.default_output_taxonomy,
          ) || data.pairs[0]
        if (pair) setWeights({ ...pair.default_weights })
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load taxonomies')
      }
    })()
  }, [])

  const pair = useMemo(() => {
    if (!tax) return null
    return (
      tax.pairs.find(
        (p) => p.input_taxonomy === inputTaxonomy && p.output_taxonomy === outputTaxonomy,
      ) || null
    )
  }, [tax, inputTaxonomy, outputTaxonomy])

  useEffect(() => {
    if (pair) setWeights({ ...pair.default_weights })
  }, [pair])

  const signals = pair?.signals || Object.keys(weights)
  const total = signals.reduce((s, k) => s + (weights[k] ?? 0), 0)
  const weightsValid = Math.abs(total - 1) <= SUM_TOLERANCE
  const thresholdValid = threshold >= 0.01 && threshold <= 0.99

  const setWeightPct = (key: string, pct: number) => {
    setWeights((prev) => ({ ...prev, [key]: Math.max(0, Math.min(1, pct / 100)) }))
  }

  const handleStart = async () => {
    setError(null)
    if (!weightsValid) {
      setError(`Weights must sum to 100% (got ${(total * 100).toFixed(1)}%)`)
      return
    }
    if (!thresholdValid) {
      setError('Threshold must be between 0.01 and 0.99')
      return
    }
    try {
      await onStart({
        model: tax?.default_model || 'P',
        input_taxonomy: inputTaxonomy,
        output_taxonomy: outputTaxonomy,
        weights,
        threshold,
        top_k: 20,
        force,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start run')
    }
  }

  if (!tax) {
    return (
      <div className="rounded-2xl border border-[var(--color-line)] bg-[var(--color-panel)]/90 px-5 py-6 text-sm text-[var(--color-mist)]">
        <Loader2 className="mr-2 inline h-4 w-4 animate-spin" />
        Loading taxonomy configuration…
      </div>
    )
  }

  return (
    <div className="space-y-5 rounded-2xl border border-[var(--color-line)] bg-[var(--color-panel)]/90 px-5 py-6">
      <div>
        <h2 className="font-display text-lg font-semibold text-[var(--color-foam)]">
          Run configuration
        </h2>
        <p className="mt-1 text-sm text-[var(--color-mist)]">
          Model P · configurable weights &amp; threshold · result reuse when config matches
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <label className="block text-xs uppercase tracking-wider text-[var(--color-mist)]">
          Model
          <select
            disabled
            value={tax.default_model}
            className="mt-1 w-full rounded-xl border border-[var(--color-line)] bg-[var(--color-panel-2)] px-3 py-2 text-sm text-[var(--color-foam)]"
          >
            {tax.models.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs uppercase tracking-wider text-[var(--color-mist)]">
          Input taxonomy
          <select
            value={inputTaxonomy}
            onChange={(e) => setInputTaxonomy(e.target.value)}
            disabled={busy}
            className="mt-1 w-full rounded-xl border border-[var(--color-line)] bg-[var(--color-panel-2)] px-3 py-2 text-sm text-[var(--color-foam)]"
          >
            {tax.taxonomies.map((t) => (
              <option key={t.key} value={t.key}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
        <label className="block text-xs uppercase tracking-wider text-[var(--color-mist)]">
          Output taxonomy
          <select
            value={outputTaxonomy}
            onChange={(e) => setOutputTaxonomy(e.target.value)}
            disabled={busy}
            className="mt-1 w-full rounded-xl border border-[var(--color-line)] bg-[var(--color-panel-2)] px-3 py-2 text-sm text-[var(--color-foam)]"
          >
            {tax.taxonomies.map((t) => (
              <option key={t.key} value={t.key}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {!pair ? (
        <p className="text-sm text-amber-300">
          No registered pair for {inputTaxonomy} → {outputTaxonomy}. Only O*NET → ESCO is
          available initially.
        </p>
      ) : null}

      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs uppercase tracking-wider text-[var(--color-mist)]">
          <span>Weights</span>
          <span className={weightsValid ? 'text-emerald-300' : 'text-amber-300'}>
            Sum {(total * 100).toFixed(1)}%
          </span>
        </div>
        {signals.map((key) => (
          <div key={key} className="grid grid-cols-[1fr_auto] items-center gap-3">
            <label className="text-sm text-[var(--color-foam)]">
              {SIGNAL_LABELS[key] || key}
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                disabled={busy}
                value={Math.round((weights[key] ?? 0) * 100)}
                onChange={(e) => setWeightPct(key, Number(e.target.value))}
                className="mt-1 w-full"
              />
            </label>
            <input
              type="number"
              min={0}
              max={100}
              step={1}
              disabled={busy}
              value={Math.round((weights[key] ?? 0) * 100)}
              onChange={(e) => setWeightPct(key, Number(e.target.value))}
              className="w-16 rounded-lg border border-[var(--color-line)] bg-[var(--color-panel-2)] px-2 py-1 text-right text-sm text-[var(--color-foam)]"
            />
          </div>
        ))}
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block text-xs uppercase tracking-wider text-[var(--color-mist)]">
          Threshold (0.01 – 0.99)
          <input
            type="number"
            min={0.01}
            max={0.99}
            step={0.01}
            disabled={busy}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="mt-1 w-full rounded-xl border border-[var(--color-line)] bg-[var(--color-panel-2)] px-3 py-2 text-sm text-[var(--color-foam)]"
          />
        </label>
        <label className="mt-6 flex items-center gap-2 text-sm text-[var(--color-foam)]">
          <input
            type="checkbox"
            checked={force}
            disabled={busy}
            onChange={(e) => setForce(e.target.checked)}
          />
          Force recompute (ignore cached run)
        </label>
      </div>

      {error ? <p className="text-sm text-red-300">{error}</p> : null}

      <button
        type="button"
        disabled={busy || !pair || !weightsValid || !thresholdValid}
        onClick={() => void handleStart()}
        className="inline-flex items-center gap-2 rounded-full bg-[var(--color-signal)] px-5 py-2.5 text-xs font-semibold uppercase tracking-wider text-[var(--color-ink)] disabled:opacity-50"
      >
        {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
        Start run
      </button>
    </div>
  )
}
