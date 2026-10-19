import { Loader2 } from 'lucide-react'
import type { RunProgressState } from './types'

const PHASE_LABELS: Record<string, string> = {
  generate_embeddings: '1. Verify Embeddings',
  calculate_similarities: '2. Calculate Similarities',
  select_top_candidates: '3. Select Top Candidates',
  apply_threshold: '4. Apply Threshold',
  run_jev_evaluation: '5. Run Jev Evaluation',
  calculate_accuracy: '6. Calculate Exact Match @1',
  save_run: '7. Save Run',
}

const DEFAULT_PHASES = [
  'generate_embeddings',
  'calculate_similarities',
  'select_top_candidates',
  'apply_threshold',
  'run_jev_evaluation',
  'calculate_accuracy',
  'save_run',
]

type Props = {
  progress: RunProgressState | null
}

export default function RunProgress({ progress }: Props) {
  if (!progress || progress.status === 'idle') return null

  const phases = progress.phases?.length ? progress.phases : DEFAULT_PHASES
  const currentIdx = progress.phase ? phases.indexOf(progress.phase) : -1
  const percent = Math.min(Math.max(progress.percent ?? 0, 0), 100)
  const running = progress.status === 'running'

  return (
    <div className="rounded-2xl border border-[var(--color-line)] bg-[var(--color-panel)]/90 px-5 py-5">
      <div className="mb-4 flex items-start gap-3">
        {running ? (
          <Loader2 className="mt-0.5 h-5 w-5 shrink-0 animate-spin text-[var(--color-signal)]" />
        ) : null}
        <div>
          <p className="font-medium text-[var(--color-foam)]">
            {progress.label || progress.run_id || 'Run'} · {progress.status}
          </p>
          <p className="text-sm text-[var(--color-mist)]">{progress.message || '…'}</p>
          {progress.error ? (
            <p className="mt-1 text-sm text-red-300">{progress.error}</p>
          ) : null}
        </div>
        <div className="ml-auto text-sm font-semibold text-[var(--color-foam)]">
          {percent.toFixed(0)}%
        </div>
      </div>
      <div className="mb-4 h-2 overflow-hidden rounded-full bg-[var(--color-panel-2)]">
        <div
          className="h-full rounded-full bg-[var(--color-signal)] transition-all"
          style={{ width: `${percent}%` }}
        />
      </div>
      <ol className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {phases.map((phase, idx) => {
          const done = currentIdx > idx || progress.status === 'completed'
          const active = currentIdx === idx && running
          return (
            <li
              key={phase}
              className={`rounded-xl border px-3 py-2 text-xs ${
                active
                  ? 'border-[var(--color-signal)] text-[var(--color-foam)]'
                  : done
                    ? 'border-[var(--color-line)] text-[var(--color-foam)]'
                    : 'border-[var(--color-line)]/50 text-[var(--color-mist)]'
              }`}
            >
              {PHASE_LABELS[phase] || phase}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
