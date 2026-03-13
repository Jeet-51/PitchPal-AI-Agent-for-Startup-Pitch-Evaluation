// PitchPal v2 - TypeScript Types

export type UserRole = "startup" | "investor";

export interface DeckQuality {
  startup_name: string;
  design_score: number;
  narrative_score: number;
  data_viz_score: number;
  overall_deck_score: number;
  design_feedback: string;
  narrative_feedback: string;
  data_viz_feedback: string;
  strengths: string[];
  improvements: string[];
  analyzed_slides: number;
}

export interface DeckUploadResponse {
  startup_name: string;
  extracted_text: string;
  slide_count: number;
  file_format: "pdf" | "pptx";
  deck_quality: DeckQuality;
}


export interface DimensionScore {
  name: string;
  score: number;
  reasoning: string;
  suggestions: string[];
  sources: string[];
  benchmark?: string;
}

export interface Contradiction {
  pitch_claim: string;
  research_finding: string;
  source?: string;
}

export interface PitchEvaluation {
  startup_name: string;
  overall_score: number;
  investment_recommendation: string;
  role: UserRole;
  dimensions: DimensionScore[];
  key_strengths: string[];
  main_concerns: string[];
  next_steps: string[];
  contradictions?: Contradiction[];
}

export interface AgentStep {
  step_number: number;
  step_type: "thought" | "action" | "observation" | "final_answer";
  content: string;
  tool_name?: string;
  tool_input?: string;
}

export interface WSMessage {
  type: "start" | "step" | "complete" | "error" | "rate_limit";
  message?: string;
  step?: AgentStep;
  evaluation?: PitchEvaluation;
  processing_time?: number;
  total_steps?: number;
  llm_provider?: string;
  cache_hits?: number;
  retry_after_seconds?: number;
  limit?: number;
  window_hours?: number;
  similar_pitch?: { startup_name: string; similarity_pct: number };
}

export interface ShareResponse {
  share_id: string;
  url: string;
  expires_in_days: number;
}

export interface SharedEvaluation {
  startup_name: string;
  role: string;
  evaluation: PitchEvaluation;
  processing_time: number;
  llm_provider: string;
  created_at: number;
  expires_at: number;
  views: number;
}

export interface SamplePitch {
  name: string;
  pitch: string;
}

export interface SavedEvaluation {
  id: string;
  timestamp: string;
  startup_name: string;
  pitch_text: string;
  evaluation: PitchEvaluation;
  processing_time: number;
  llm_provider: string;
  role: UserRole;
}
