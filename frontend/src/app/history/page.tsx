"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import Header from "@/components/Header";
import EvaluationResults from "@/components/EvaluationResults";
import { SavedEvaluation, UserRole } from "@/types";
import { getHistory, deleteEvaluation, clearHistory } from "@/lib/storage";
import { getRole } from "@/lib/auth";
import { clearBackendCache, deleteCacheEntry } from "@/lib/api";
import {
  Trash2, ChevronDown, BarChart3,
  TrendingUp, Award, ThumbsUp, History, FileText,
} from "lucide-react";

function scoreColor(s: number) {
  if (s >= 7) return "var(--success)";
  if (s >= 5) return "var(--warning)";
  return "var(--danger)";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const fadeUp: any = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1, y: 0,
    transition: { duration: 0.4, delay: i * 0.06 },
  }),
};

export default function HistoryPage() {
  const router = useRouter();
  const [history, setHistory] = useState<SavedEvaluation[]>([]);
  const [expandedId, setExpanded] = useState<string | null>(null);
  const [pitchOpenId, setPitchOpen] = useState<string | null>(null);
  const [role, setRole] = useState<UserRole>("startup");

  useEffect(() => {
    const r = getRole();
    if (!r) { router.push("/"); return; }
    setRole(r);
    setHistory(getHistory(r));
  }, [router]);

  const handleDelete = (id: string) => {
    // Find the entry BEFORE removing it so we can clear its backend cache
    const entry = history.find((e) => e.id === id);
    deleteEvaluation(id, role);
    setHistory(getHistory(role));
    if (expandedId === id) setExpanded(null);
    if (pitchOpenId === id) setPitchOpen(null);
    // Also remove from backend evaluation cache so re-submitting runs fresh
    if (entry) {
      deleteCacheEntry(entry.pitch_text ?? "", role);
    }
  };

  const handleClear = () => {
    if (!confirm("Clear all evaluation history?")) return;
    clearHistory(role);
    clearBackendCache();
    setHistory([]);
    setExpanded(null);
  };

  /* Stats */
  const total = history.length;
  const avg = total ? history.reduce((s, e) => s + e.evaluation.overall_score, 0) / total : 0;
  const best = total ? history.reduce((b, e) => e.evaluation.overall_score > b.evaluation.overall_score ? e : b) : null;
  const buys = history.filter(e => e.evaluation.investment_recommendation.toLowerCase().includes("buy")).length;

  const stats = [
    { label: "Total Evaluations", value: total, icon: BarChart3, color: "var(--primary)" },
    { label: "Average Score", value: avg.toFixed(1), icon: TrendingUp, color: scoreColor(avg) },
    { label: "Best Pitch", value: best?.startup_name ?? "—", icon: Award, color: "var(--warning)" },
    { label: "Buy Recommendations", value: buys, icon: ThumbsUp, color: "var(--success)" },
  ];

  return (
    <div className="min-h-screen bg-[var(--background)]">
      <div className="fixed inset-0 pointer-events-none -z-10 overflow-hidden">
        <div className="orb orb-1 absolute w-[480px] h-[480px] -top-40 -right-24 opacity-50" />
        <div className="orb orb-2 absolute w-[360px] h-[360px] -bottom-24 -left-24 opacity-40" />
      </div>

      <Header />

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
        {/* Page header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: "linear-gradient(135deg,var(--primary),var(--accent))" }}>
              <History className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-[var(--foreground)]">Evaluation History</h2>
              <p className="text-sm text-[var(--muted)] mt-0.5">
                {total} evaluation{total !== 1 ? "s" : ""} ·{" "}
                <span className={`font-medium px-1.5 py-0.5 rounded-full text-xs ${role === "investor"
                  ? "bg-purple-500/10 text-purple-400"
                  : "bg-blue-500/10 text-blue-400"
                  }`}>
                  {role === "investor" ? "Investor" : "Startup"} Mode
                </span>
              </p>
            </div>
          </div>

          {total > 0 && (
            <button
              onClick={handleClear}
              className="flex items-center gap-2 px-4 py-2 text-sm rounded-xl glass border border-[var(--danger)]/30 text-[var(--danger)] hover:bg-[var(--danger)]/10 transition-colors"
            >
              <Trash2 className="w-4 h-4" /> Clear All
            </button>
          )}
        </div>

        {/* Stats */}
        {total > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {stats.map((s, i) => (
              <motion.div
                key={s.label}
                custom={i}
                variants={fadeUp}
                initial="hidden"
                animate="visible"
                className="glass-card rounded-2xl p-4 text-center"
              >
                <s.icon className="w-5 h-5 mx-auto mb-2" style={{ color: s.color }} />
                <p className="text-xl font-bold text-[var(--foreground)] truncate">{s.value}</p>
                <p className="text-[10px] text-[var(--muted)] mt-0.5 uppercase tracking-wide">{s.label}</p>
              </motion.div>
            ))}
          </div>
        )}

        {/* History list */}
        {total === 0 ? (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="glass-card rounded-2xl p-14 text-center">
            <BarChart3 className="w-12 h-12 text-[var(--muted)] mx-auto mb-4" />
            <h3 className="font-semibold text-[var(--foreground)] mb-2">No evaluations yet</h3>
            <p className="text-sm text-[var(--muted)]">Submit a pitch to start building your history.</p>
          </motion.div>
        ) : (
          <div className="space-y-3">
            {history.map((entry, i) => (
              <motion.div
                key={entry.id}
                custom={i}
                variants={fadeUp}
                initial="hidden"
                animate="visible"
                className="glass-card rounded-2xl overflow-hidden"
              >
                {/* Row */}
                <div
                  className="flex items-center justify-between px-5 py-4 cursor-pointer hover:bg-[var(--hover-bg)] transition-colors"
                  onClick={() => setExpanded(expandedId === entry.id ? null : entry.id)}
                >
                  <div className="flex items-center gap-4">
                    <span className="text-2xl font-black tabular-nums"
                      style={{ color: scoreColor(entry.evaluation.overall_score) }}>
                      {entry.evaluation.overall_score.toFixed(1)}
                    </span>
                    <div>
                      <h4 className="font-semibold text-[var(--foreground)]">{entry.startup_name}</h4>
                      <p className="text-xs text-[var(--muted)]">
                        {new Date(entry.timestamp).toLocaleDateString("en-US", {
                          month: "short", day: "numeric", year: "numeric",
                          hour: "2-digit", minute: "2-digit",
                        })} · {entry.processing_time.toFixed(1)}s · {entry.evaluation.investment_recommendation}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(entry.id); }}
                      className="p-1.5 rounded-lg text-[var(--muted)] hover:text-[var(--danger)] hover:bg-[var(--danger)]/10 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                    <motion.div animate={{ rotate: expandedId === entry.id ? 180 : 0 }} transition={{ duration: 0.2 }}>
                      <ChevronDown className="w-5 h-5 text-[var(--muted)]" />
                    </motion.div>
                  </div>
                </div>

                {/* Expanded */}
                <AnimatePresence initial={false}>
                  {expandedId === entry.id && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="overflow-hidden"
                    >
                      <div className="border-t border-[var(--card-border)] px-5 pt-4 pb-5 space-y-4">

                        {/* ── View Pitch toggle ── */}
                        <div>
                          <button
                            onClick={() => setPitchOpen(pitchOpenId === entry.id ? null : entry.id)}
                            className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all"
                            style={{
                              background: pitchOpenId === entry.id
                                ? "var(--accent-primary)"
                                : "var(--hover-bg)",
                              color: pitchOpenId === entry.id
                                ? "#fff"
                                : "var(--muted)",
                              border: "1px solid var(--card-border)",
                            }}
                          >
                            <FileText className="w-3.5 h-3.5" />
                            {pitchOpenId === entry.id ? "Hide Pitch" : "View Original Pitch"}
                            <motion.span
                              animate={{ rotate: pitchOpenId === entry.id ? 180 : 0 }}
                              transition={{ duration: 0.2 }}
                              style={{ display: "inline-flex" }}
                            >
                              <ChevronDown className="w-3.5 h-3.5" />
                            </motion.span>
                          </button>

                          <AnimatePresence initial={false}>
                            {pitchOpenId === entry.id && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: "auto", opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.25 }}
                                className="overflow-hidden"
                              >
                                <div
                                  className="mt-3 rounded-xl p-4 text-sm leading-relaxed text-[var(--foreground)] whitespace-pre-wrap max-h-64 overflow-y-auto"
                                  style={{
                                    background: "var(--hover-bg)",
                                    border: "1px solid var(--card-border)",
                                    fontFamily: "var(--font-mono, monospace)",
                                    fontSize: "0.8rem",
                                    lineHeight: "1.7",
                                  }}
                                >
                                  {entry.pitch_text || "Pitch text not available."}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>

                        {/* ── Evaluation results ── */}
                        <EvaluationResults
                          evaluation={entry.evaluation}
                          processingTime={entry.processing_time}
                          llmProvider={entry.llm_provider}
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
