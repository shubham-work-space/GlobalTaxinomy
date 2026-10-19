import { BarChart3, GitCompareArrows } from 'lucide-react'
import { useState } from 'react'
import CrosswalkPage from './CrosswalkPage'
import EvaluationPage from './EvaluationPage'
import type { AppTab } from './types'

const TABS: { id: AppTab; label: string; icon: typeof GitCompareArrows }[] = [
  { id: 'crosswalk', label: 'Crosswalk', icon: GitCompareArrows },
  { id: 'evaluation', label: 'Evals', icon: BarChart3 },
]

export default function App() {
  const [tab, setTab] = useState<AppTab>('crosswalk')

  return (
    <div className="grid-noise min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-5 py-8 md:px-8 md:py-12">
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <p className="font-display text-sm font-semibold tracking-wide text-[var(--color-foam)]">
            Taxonomy Crosswalk 4.0
          </p>
          <div className="inline-flex rounded-full border border-[var(--color-line)] bg-[var(--color-panel)]/80 p-1">
            {TABS.map((t) => {
              const Icon = t.icon
              const active = tab === t.id
              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setTab(t.id)}
                  className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] transition ${
                    active
                      ? 'bg-[var(--color-signal)] text-[var(--color-ink)]'
                      : 'text-[var(--color-mist)] hover:text-[var(--color-foam)]'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {t.label}
                </button>
              )
            })}
          </div>
        </div>

        <div className={tab === 'crosswalk' ? 'block' : 'hidden'}>
          <CrosswalkPage />
        </div>
        <div className={tab === 'evaluation' ? 'block' : 'hidden'}>
          <EvaluationPage />
        </div>
      </div>
    </div>
  )
}
