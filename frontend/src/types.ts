export type DatasetProgress = {
  total: number
  completed: number
  failed: number
  skipped?: number
  current_folder: string | null
  files_in_folder: number
  files_done: number
  success: number
  pending: number
  failed_requests: number
  percent: number
  status: string
  errors: string[]
}

export type VariantProgress = {
  status: string
  total: number
  completed: number
  percent: number
  current: string | null
  items_done: number
  items_total: number
  message: string
  counts: Record<string, number>
}

export type JobState = {
  status: 'idle' | 'running' | 'completed' | 'failed'
  mode: 'create' | 'refresh' | 'variants' | null
  embeddings_exist: boolean
  started_at: string | null
  finished_at: string | null
  onet: DatasetProgress
  esco: DatasetProgress
  variants?: VariantProgress
  summary: {
    total_embeddings: number
    total_failures: number
    onet_embeddings: number
    esco_embeddings: number
    variant_embeddings?: number
  }
  message: string
  version?: number
}

export type DatasetInventory = {
  occupations: number
  vectors: number
  facets: {
    title: number
    description: number
    skills: number
    knowledge: number
  }
}

export type EmbeddingInventory = {
  exists: boolean
  total_occupations: number
  total_vectors: number
  model: string
  created_at: string | null
  store: string
  path: string
  datasets: {
    onet: DatasetInventory
    esco: DatasetInventory
  }
  collections: Record<string, number>
}

export type StatusResponse = {
  embeddings_exist: boolean
  button_label: string
  default_mode: 'create' | 'refresh'
  store?: string
  counts?: {
    onet: number
    esco: number
  }
  inventory?: EmbeddingInventory
  variant_inventory?: Record<string, number>
  variant_embeddings_exist?: boolean
  variant_partial?: boolean
  variant_ready_collections?: number
  variant_total_collections?: number
  job: JobState
}

export type EvalProgress = {
  status: string
  phase: string | null
  started_at: string | null
  finished_at: string | null
  message: string
  done: number
  total: number
  percent: number
  current?: string | null
  error?: string | null
}

export type AppTab = 'crosswalk' | 'evaluation'

export type TaxonomyPillar = {
  key: string
  collection_suffix: string
  aggregate: string | null
}

export type TaxonomyInfo = {
  key: string
  label: string
  pillars: TaxonomyPillar[]
  id_field: string
  label_collection: string | null
}

export type TaxonomyPairInfo = {
  direction: string
  input_taxonomy: string
  output_taxonomy: string
  label: string
  signals: string[]
  default_weights: Record<string, number>
  ground_truth_path: string
}

export type TaxonomiesResponse = {
  default_model: string
  default_input_taxonomy: string
  default_output_taxonomy: string
  default_direction: string
  taxonomies: TaxonomyInfo[]
  pairs: TaxonomyPairInfo[]
  models: { key: string; label: string; description: string }[]
}

export type RunConfig = {
  model: string
  input_taxonomy: string
  output_taxonomy: string
  weights: Record<string, number>
  threshold: number
  top_k?: number
  force?: boolean
}

export type RunMetrics = {
  total_records?: number | null
  records_sent_to_jev?: number | null
  records_sent_to_llm?: number | null
  exact_match_at_1?: number | null
  accuracy_at_1?: number | null
  accuracy_at_2?: number | null
  accuracy_at_3?: number | null
  accuracy_at_5?: number | null
  accuracy_at_10?: number | null
  embedding_top1_accuracy?: number | null
  final_accuracy_after_llm?: number | null
  llm_subset_accuracy_before?: number | null
  llm_subset_accuracy_after?: number | null
  jev_subset_accuracy_after?: number | null
  accuracy_gain?: number | null
  resolver?: string | null
  high_confidence_records?: number | null
  high_confidence_correct?: number | null
  high_confidence_accuracy?: number | null
  resolver_subset_records?: number | null
  resolver_subset_correct?: number | null
  resolver_subset_accuracy?: number | null
}

export type RunSummary = {
  run_id: string
  label: string
  seq: number
  fingerprint: string
  model: string
  input_taxonomy: string
  output_taxonomy: string
  direction: string
  weights: Record<string, number>
  threshold: number
  status: string
  created_at: string | null
  finished_at: string | null
  phase?: string | null
  percent?: number
  message?: string
  reused_from?: string | null
  seed?: string
  metrics_summary?: RunMetrics | null
  metrics?: RunMetrics
  config?: RunConfig
}

export type RunRecord = {
  source_id: string
  source_title?: string
  top1_target_id?: string | null
  top1_preferred_title?: string | null
  top1_similarity?: number | null
  sent_to_llm?: boolean
  llm_target_id?: string | null
  llm_preferred_title?: string | null
  llm_match_label?: string | null
  final_target_id?: string | null
  final_preferred_title?: string | null
  top1_exact_hit?: boolean
  top1_any_hit?: boolean
  final_exact_hit?: boolean
  final_any_hit?: boolean
  ground_truth_ids?: string[]
  candidates?: Array<{
    rank: number
    target_id: string
    preferred_title?: string
    similarity?: number
  }>
}

export type RunProgressState = {
  status: 'idle' | 'running' | 'completed' | 'failed'
  run_id: string | null
  label: string | null
  phase: string | null
  phases?: string[]
  phase_index?: number
  percent: number
  message: string
  done?: number
  total?: number
  error?: string | null
  version?: number
}

export type OptimizeBestSoFar = {
  accuracy: number
  weights: Record<string, number>
  mode?: string
}

export type OptimizeJobState = {
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled'
  phase: string | null
  started_at: string | null
  finished_at: string | null
  message: string
  done: number
  total: number
  percent: number
  best_so_far: OptimizeBestSoFar | null
  config: Record<string, unknown> | null
  result: OptimizeResult | null
  error: string | null
  version?: number
}

export type OptimizeLeaderboardEntry = {
  rank: number
  weights: Record<string, number>
  mode?: string
  accuracy: number
  margin: number
}

export type OptimizeResult = {
  timestamp?: string
  run_id: string
  variant?: string
  direction?: string
  metric: string
  coarse_step?: number
  refine_step?: number
  n_sources?: number
  n_targets?: number
  note?: string
  baseline: {
    weights: Record<string, number>
    mode?: string
    accuracy_at_1: number
    topk?: Record<string, number>
    cv_mean?: number
    cv_folds?: number[]
    margin?: number
  }
  optimized: {
    weights: Record<string, number>
    mode?: string
    accuracy_at_1: number
    topk?: Record<string, number>
    cv_mean?: number
    cv_folds?: number[]
    margin?: number
    delta_accuracy: number
    delta_cv?: number
  }
  /** Present for full P-grid runs; omitted for lightweight V6 / fixed-weight evals. */
  significance?: {
    mcnemar_p_value: number
    significant_at_05: boolean
    interpretation: string
  }
  plateau_size?: number
  candidates_evaluated?: number
  leaderboard?: OptimizeLeaderboardEntry[]
}

export type OptimizeStatusResponse = {
  job: OptimizeJobState
  active_weights: Record<string, number>
  weights_metadata: Record<string, unknown>
  latest: {
    run_id: string
    run_path: string
    sensitivity_path: string
    errors_path: string
    result: OptimizeResult
  } | null
}

export type WeightSource = 'baseline' | 'manual' | 'optimized' | 'variant' | 'ejento'

export type TopKMetrics = Record<string, number>

export type WeightRun = {
  run_id: string
  source: WeightSource
  label: string
  variant?: string
  weights: Record<string, number>
  applied: boolean
  created_at: string
  occupations_exact: number
  occupations_any: number
  exact: TopKMetrics
  any: TopKMetrics
  final_score: number | null
  margin: number | null
  cached?: boolean
}

export type ApproachGroup = {
  baseline: WeightRun | null
  optimized: WeightRun | null
  ejento?: WeightRun | null
  manual: WeightRun[]
  latest_manual: WeightRun | null
}

export type WeightRunsResponse = {
  runs: WeightRun[]
  grouped: ApproachGroup
  variant_runs?: WeightRun[]
  active_weights: Record<string, number>
  weights_metadata: Record<string, unknown>
  matrix_ready: boolean
  default_weights: Record<string, number>
  run?: WeightRun
}

export type WeightEvaluateBody = {
  weights: Record<string, number>
  source: WeightSource
  label?: string
  apply?: boolean
  variant?: string
}

export type CrosswalkDirection = 'onet_to_esco' | 'esco_to_onet'

export type CrosswalkStep = 'embedding' | 'agent'

export type GroundTruthMatchType =
  | 'exactMatch'
  | 'closeMatch'
  | 'broadMatch'
  | 'narrowMatch'

export type GroundTruthTypeCounts = Record<GroundTruthMatchType, number>

export type OccupationSuggestion = {
  occupation_id: string
  preferred_title: string
}

export type CrosswalkMatch = {
  occupation_id: string
  preferred_title: string
  weighted_score: number
  title_similarity: number | null
  description_similarity: number | null
  skills_similarity: number | null
  knowledge_similarity: number | null
  facet_scores?: {
    title: number | null
    description: number | null
    skills: number | null
    knowledge: number | null
  }
  facets_used?: string[]
  weights_applied?: Record<string, number>
  ground_truth_match_type?: GroundTruthMatchType | null
  ground_truth_match_type_official?: GroundTruthMatchType | null
}

export type CrosswalkStageResult = {
  step: string
  match_count: number
  matches: CrosswalkMatch[]
  ground_truth_total: number
  ground_truth_found_in_top_k: number
  ground_truth_found_by_type?: Partial<GroundTruthTypeCounts>
  ground_truth_expected_by_type?: Partial<GroundTruthTypeCounts>
}

export type GroundTruthSourceSummary = {
  total: number
  by_type: Partial<GroundTruthTypeCounts>
  has_exact_match: boolean
  exact_match_count: number
}

export type GroundTruthCardinality = {
  direction: CrosswalkDirection
  source_count: number
  mean_targets_per_source: number
  median_targets_per_source: number
  max_targets_per_source: number
  percent_single_target: number
  sources_with_exact_match: number
  sources_with_any_match: number
  inverted_from_official: boolean
}

export type ApiErrorDetail = {
  message: string
  error_type?: string
  step?: string
  context?: Record<string, unknown>
  cause?: string
  traceback?: string
}

export type RankingSource = 'embedding' | 'jev' | 'jev_cached'

export type CrosswalkResult = {
  direction: CrosswalkDirection
  source_dataset: 'onet' | 'esco'
  target_dataset: 'onet' | 'esco'
  source: {
    occupation_id: string
    preferred_title: string
    dataset: 'onet' | 'esco'
  }
  weights: Record<string, number>
  top_k: number
  candidate_pool?: number
  pool_size?: number
  threshold?: number
  top1_similarity?: number
  ranking_source?: RankingSource
  sent_to_jev?: boolean
  matches?: CrosswalkMatch[]
  ground_truth?: {
    source: GroundTruthSourceSummary
    cardinality: GroundTruthCardinality
    inverted_from_official: boolean
  }
  embedding: CrosswalkStageResult
  ranked?: CrosswalkStageResult
  jev?: {
    cache_origin?: string | null
    jev_ranked_target_ids?: string[]
    jev_error?: string | null
    jev_match_label?: string | null
    jev_confidence?: number | null
  } | null
}
