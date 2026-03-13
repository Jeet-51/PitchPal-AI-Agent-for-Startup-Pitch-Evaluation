"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence, useMotionValue, useTransform, animate } from "framer-motion";
import {
  ChevronDown, Download, TrendingUp, AlertCircle,
  ArrowRight, Clock, Cpu, Database, Star,
  ExternalLink, ShieldAlert,
} from "lucide-react";
import { PitchEvaluation } from "@/types";
import RadarChart from "./RadarChart";
import ScoreBarChart from "./ScoreBarChart";
import { exportPitchPDF } from "@/lib/pdfExport";
import ShareButton from "./ShareButton";

interface Props {
  evaluation: PitchEvaluation;
  processingTime?: number;
  llmProvider?: string;
  cacheHits?: number;
  role?: string;
  readOnly?: boolean;  // hides share button (used on the shared eval page itself)
}

/* ── Helpers ──────────────────────────────────────────────── */
function scoreColor(s: number) {
  if (s >= 7) return "var(--success)";
  if (s >= 5) return "var(--warning)";
  return "var(--danger)";
}

function recStyle(rec: string): { bg: string; color: string; border: string } {
  const r = rec.toLowerCase();
  if (r.includes("strong buy")) return { bg: "rgba(52,211,153,0.12)", color: "#34d399", border: "rgba(52,211,153,0.3)" };
  if (r.includes("buy")) return { bg: "rgba(16,185,129,0.12)", color: "#10b981", border: "rgba(16,185,129,0.3)" };
  if (r.includes("hold")) return { bg: "rgba(251,191,36,0.12)", color: "#fbbf24", border: "rgba(251,191,36,0.3)" };
  if (r.includes("strong pass")) return { bg: "rgba(239,68,68,0.15)", color: "#f87171", border: "rgba(239,68,68,0.35)" };
  if (r.includes("pass")) return { bg: "rgba(248,113,113,0.12)", color: "#f87171", border: "rgba(248,113,113,0.3)" };
  return { bg: "rgba(129,140,248,0.12)", color: "var(--primary)", border: "rgba(129,140,248,0.25)" };
}

/* ── Animated score number ────────────────────────────────── */
function AnimatedScore({ target }: { target: number }) {
  const count = useMotionValue(0);
  const display = useTransform(count, (v) => v.toFixed(1));

  useEffect(() => {
    const ctrl = animate(count, target, { duration: 1.4, ease: "easeOut" });
    return ctrl.stop;
  }, [target, count]);

  return (
    <motion.span style={{ fontSize: "inherit", fontWeight: "inherit", color: scoreColor(target) }}>
      {display}
    </motion.span>
  );
}

/* ── Dimension card ───────────────────────────────────────── */
function DimCard({ dim, index }: { dim: PitchEvaluation["dimensions"][0]; index: number }) {
  const [open, setOpen] = useState(false);
  const pct = (dim.score / 10) * 100;
  const color = scoreColor(dim.score);

  // Extract clean domain label from URL
  const sourceDomain = (url: string) => {
    try { return new URL(url).hostname.replace("www.", ""); }
    catch { return url; }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.06, ease: [0.22, 1, 0.36, 1] }}
      className="glass-card rounded-xl overflow-hidden"
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full px-4 py-3.5 text-left hover:bg-[var(--hover-bg)] transition-colors"
      >
        <div className="flex items-center justify-between gap-3 mb-2">
          <span className="text-sm font-semibold text-[var(--foreground)] capitalize">
            {dim.name.replace(/_/g, " ")}
          </span>
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-sm font-bold" style={{ color }}>{dim.score.toFixed(1)}</span>
            <motion.div animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
              <ChevronDown className="w-4 h-4 text-[var(--muted)]" />
            </motion.div>
          </div>
        </div>

        {/* Score bar */}
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--card-border)" }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 1, delay: index * 0.06 + 0.2, ease: "easeOut" }}
            className="h-full rounded-full"
            style={{ background: color }}
          />
        </div>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1 border-t border-[var(--card-border)] space-y-3">
              <p className="text-sm text-[var(--muted)] leading-relaxed">{dim.reasoning}</p>

              {/* Benchmark */}
              {dim.benchmark && (
                <div
                  className="text-xs px-3 py-2 rounded-lg"
                  style={{ background: "rgba(129,140,248,0.08)", borderLeft: "2px solid var(--primary)" }}
                >
                  <span className="font-semibold text-[var(--primary)]">Benchmark: </span>
                  <span className="text-[var(--foreground)]">{dim.benchmark}</span>
                </div>
              )}

              {dim.suggestions?.length > 0 && (
                <div className="space-y-1.5">
                  <p className="text-xs font-semibold uppercase tracking-widest text-[var(--muted)]">Suggestions</p>
                  {dim.suggestions.map((s, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm text-[var(--foreground)]">
                      <ArrowRight className="w-3.5 h-3.5 shrink-0 mt-0.5 text-[var(--primary)]" />
                      {s}
                    </div>
                  ))}
                </div>
              )}

              {/* Sources */}
              {dim.sources && dim.sources.length > 0 && (
                <div className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-widest text-[var(--muted)]">Sources</p>
                  <div className="flex flex-wrap gap-2">
                    {dim.sources.map((url, i) => (
                      <a
                        key={i}
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-md transition-colors hover:opacity-80"
                        style={{ background: "var(--hover-bg)", color: "var(--primary)", border: "1px solid rgba(129,140,248,0.2)" }}
                      >
                        <ExternalLink className="w-2.5 h-2.5" />
                        {sourceDomain(url)}
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

/* ── Main component ───────────────────────────────────────── */
export default function EvaluationResults({
  evaluation, processingTime, llmProvider, cacheHits = 0, role, readOnly = false
}: Props) {
  const [chartView, setChartView] = useState<"radar" | "bar">("radar");
  const rec = recStyle(evaluation.investment_recommendation);

  return (
    <div className="space-y-6">

      {/* ── Score hero ──────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.45 }}
        className="flex items-center justify-between gap-4 flex-wrap"
      >
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-[var(--muted)] mb-1">Overall Score</p>
          <div className="text-5xl font-black">
            <AnimatedScore target={evaluation.overall_score} />
            <span className="text-2xl font-medium text-[var(--muted)]"> / 10</span>
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <span
            className="px-4 py-2 rounded-xl text-sm font-bold"
            style={{ background: rec.bg, color: rec.color, border: `1px solid ${rec.border}` }}
          >
            {evaluation.investment_recommendation}
          </span>

          <div className="flex items-center gap-2">
            {/* Share button */}
            {!readOnly && (
              <ShareButton
                evaluation={evaluation}
                startupName={evaluation.startup_name}
                role={role || evaluation.role || "startup"}
                processingTime={processingTime}
                llmProvider={llmProvider}
              />
            )}
            {/* PDF export */}
            <button
              onClick={() => exportPitchPDF(evaluation, processingTime, llmProvider)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--hover-bg)] transition-colors"
            >
              <Download className="w-3.5 h-3.5" /> Export PDF
            </button>
          </div>
        </div>
      </motion.div>

      {/* ── Charts ──────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.1 }}
        className="glass-card rounded-xl overflow-hidden"
      >
        {/* Chart toggle */}
        <div className="flex border-b border-[var(--card-border)]">
          {(["radar", "bar"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setChartView(v)}
              className="flex-1 py-2.5 text-sm font-medium capitalize transition-colors relative"
              style={{
                color: chartView === v ? "var(--primary)" : "var(--muted)",
                background: chartView === v ? "rgba(129,140,248,0.07)" : "transparent",
              }}
            >
              {v === "radar" ? "Radar Chart" : "Score Bars"}
              {chartView === v && (
                <motion.div
                  layoutId="chart-tab"
                  className="absolute bottom-0 inset-x-0 h-0.5"
                  style={{ background: "var(--primary)" }}
                />
              )}
            </button>
          ))}
        </div>
        <div className="p-4">
          <AnimatePresence mode="wait">
            <motion.div
              key={chartView}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.22 }}
            >
              {chartView === "radar"
                ? <RadarChart evaluation={evaluation} />
                : <ScoreBarChart evaluation={evaluation} />}
            </motion.div>
          </AnimatePresence>
        </div>
      </motion.div>

      {/* ── Dimension cards ─────────────────────────────────── */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-[var(--muted)] mb-3">
          Dimension Breakdown
        </p>
        <div className="space-y-2">
          {evaluation.dimensions.map((dim, i) => (
            <DimCard key={dim.name} dim={dim} index={i} />
          ))}
        </div>
      </div>

      {/* ── Contradictions panel ─────────────────────────────── */}
      {evaluation.contradictions && evaluation.contradictions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
          className="rounded-xl p-4 space-y-3"
          style={{ background: "rgba(251,146,60,0.07)", border: "1px solid rgba(251,146,60,0.25)" }}
        >
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4" style={{ color: "#fb923c" }} />
            <h4 className="text-sm font-bold" style={{ color: "#fb923c" }}>
              Fact-Check Findings
            </h4>
            <span
              className="ml-auto text-xs px-2 py-0.5 rounded-full font-semibold"
              style={{ background: "rgba(251,146,60,0.15)", color: "#fb923c" }}
            >
              {evaluation.contradictions.length} claim{evaluation.contradictions.length !== 1 ? "s" : ""} flagged
            </span>
          </div>
          <p className="text-xs text-[var(--muted)]">
            PitchPal found research that contradicts these claims from your pitch:
          </p>
          <div className="space-y-3">
            {evaluation.contradictions.map((c, i) => (
              <div
                key={i}
                className="rounded-lg p-3 space-y-1.5"
                style={{ background: "rgba(0,0,0,0.2)", border: "1px solid rgba(251,146,60,0.15)" }}
              >
                <div className="flex items-start gap-2">
                  <span className="text-xs font-bold shrink-0 mt-0.5" style={{ color: "#fb923c" }}>Claimed:</span>
                  <span className="text-xs text-[var(--foreground)] italic">&quot;{c.pitch_claim}&quot;</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-xs font-bold shrink-0 mt-0.5" style={{ color: "#34d399" }}>Found:</span>
                  <span className="text-xs text-[var(--muted)]">{c.research_finding}</span>
                </div>
                {c.source && (
                  <div className="flex items-center gap-1 pt-0.5">
                    <ExternalLink className="w-2.5 h-2.5 text-[var(--muted)]" />
                    <span className="text-[10px] text-[var(--muted)] opacity-70">{c.source}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* ── Insights ────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4">
        {/* Strengths */}
        <motion.div
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="rounded-xl p-4"
          style={{ background: "rgba(52,211,153,0.07)", border: "1px solid rgba(52,211,153,0.2)" }}
        >
          <div className="flex items-center gap-2 mb-3">
            <Star className="w-4 h-4 text-[var(--success)]" />
            <h4 className="text-sm font-bold text-[var(--success)]">Key Strengths</h4>
          </div>
          <ul className="space-y-1.5">
            {evaluation.key_strengths.map((s, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-[var(--foreground)]">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--success)] mt-1.5 shrink-0" />
                {s}
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Concerns */}
        <motion.div
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.25 }}
          className="rounded-xl p-4"
          style={{ background: "rgba(251,191,36,0.07)", border: "1px solid rgba(251,191,36,0.2)" }}
        >
          <div className="flex items-center gap-2 mb-3">
            <AlertCircle className="w-4 h-4 text-[var(--warning)]" />
            <h4 className="text-sm font-bold text-[var(--warning)]">Main Concerns</h4>
          </div>
          <ul className="space-y-1.5">
            {evaluation.main_concerns.map((c, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-[var(--foreground)]">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--warning)] mt-1.5 shrink-0" />
                {c}
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Next steps */}
        <motion.div
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
          className="rounded-xl p-4"
          style={{ background: "rgba(129,140,248,0.07)", border: "1px solid rgba(129,140,248,0.2)" }}
        >
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-[var(--primary)]" />
            <h4 className="text-sm font-bold text-[var(--primary)]">Next Steps</h4>
          </div>
          <ol className="space-y-1.5">
            {evaluation.next_steps.map((n, i) => (
              <li key={i} className="flex items-start gap-2.5 text-sm text-[var(--foreground)]">
                <span
                  className="w-5 h-5 rounded-full text-xs font-bold flex items-center justify-center shrink-0 mt-0.5"
                  style={{ background: "rgba(129,140,248,0.15)", color: "var(--primary)" }}
                >
                  {i + 1}
                </span>
                {n}
              </li>
            ))}
          </ol>
        </motion.div>
      </div>

      {/* ── Meta ────────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-2 pt-1">
        {processingTime && (
          <span className="flex items-center gap-1.5 px-2.5 py-1 glass rounded-lg text-xs text-[var(--muted)]">
            <Clock className="w-3 h-3" /> {processingTime.toFixed(1)}s
          </span>
        )}
        {llmProvider && (
          <span className="flex items-center gap-1.5 px-2.5 py-1 glass rounded-lg text-xs text-[var(--muted)]">
            <Cpu className="w-3 h-3" /> {llmProvider}
          </span>
        )}
        {cacheHits > 0 && (
          <span className="flex items-center gap-1.5 px-2.5 py-1 glass rounded-lg text-xs text-[var(--success)]">
            <Database className="w-3 h-3" /> {cacheHits} cache hit{cacheHits !== 1 ? "s" : ""}
          </span>
        )}
      </div>
    </div>
  );
}
