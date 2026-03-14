"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import Header from "@/components/Header";
import { SavedEvaluation, UserRole } from "@/types";
import { getHistory } from "@/lib/storage";
import { getRole } from "@/lib/auth";
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer, Tooltip, Legend,
} from "recharts";
import { GitCompare, CheckCircle2, Circle } from "lucide-react";

const COLORS = ["#818cf8", "#34d399", "#f87171"];

function scoreColor(s: number) {
  if (s >= 7) return "var(--success)";
  if (s >= 5) return "var(--warning)";
  return "var(--danger)";
}

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  visible: (i: number) => ({
    opacity: 1, y: 0,
    transition: { duration: 0.38, delay: i * 0.07 },
  }),
};

export default function ComparePage() {
  const router = useRouter();
  const [history, setHistory] = useState<SavedEvaluation[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [role, setRole] = useState<UserRole>("startup");

  useEffect(() => {
    const r = getRole();
    if (!r) { router.push("/"); return; }
    setRole(r);
    setHistory(getHistory(r));
  }, [router]);

  const toggle = (id: string) =>
    setSelected((p) =>
      p.includes(id) ? p.filter((s) => s !== id) : p.length >= 3 ? p : [...p, id]
    );

  const picks = history.filter((e) => selected.includes(e.id));
  const dimNames = picks[0]?.evaluation.dimensions.map((d) => d.name) ?? [];

  const radarData = dimNames.map((name, di) => {
    const row: Record<string, string | number> = {
      dim: name.length > 14 ? name.slice(0, 13) + "…" : name,
    };
    picks.forEach((e) => { row[e.startup_name] = e.evaluation.dimensions[di]?.score ?? 0; });
    return row;
  });

  return (
    <div className="min-h-screen bg-[var(--background)]">
      <div className="fixed inset-0 pointer-events-none -z-10 overflow-hidden">
        <div className="orb orb-1 absolute w-[480px] h-[480px] -top-40 -right-24 opacity-50" />
        <div className="orb orb-2 absolute w-[360px] h-[360px] -bottom-24 -left-24 opacity-40" />
      </div>

      <Header />

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
        {/* Page header */}
        <div className="mb-8 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: "linear-gradient(135deg,var(--primary),var(--accent))" }}>
            <GitCompare className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-[var(--foreground)]">Compare Pitches</h2>
            <p className="text-sm text-[var(--muted)] mt-0.5">
              Select up to 3 evaluations to compare side-by-side ·{" "}
              <span className={`font-medium px-1.5 py-0.5 rounded-full text-xs ${role === "investor" ? "bg-purple-500/10 text-purple-400" : "bg-blue-500/10 text-blue-400"
                }`}>
                {role === "investor" ? "Investor" : "Startup"} Mode
              </span>
            </p>
          </div>
        </div>

        {history.length === 0 ? (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="glass-card rounded-2xl p-14 text-center">
            <GitCompare className="w-12 h-12 text-[var(--muted)] mx-auto mb-4" />
            <h3 className="font-semibold text-[var(--foreground)] mb-2">No evaluations to compare</h3>
            <p className="text-sm text-[var(--muted)]">Evaluate at least 2 startup pitches first.</p>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* ── Left: Selector ─────────────────────────────── */}
            <div className="lg:col-span-1">
              <div className="glass-card rounded-2xl p-4 sticky top-20">
                <h3 className="text-sm font-bold text-[var(--foreground)] mb-3">
                  Select Pitches{" "}
                  <span className="font-normal text-[var(--muted)]">({selected.length}/3)</span>
                </h3>
                <div className="space-y-2 max-h-[500px] overflow-y-auto pr-1">
                  {history.map((entry, i) => {
                    const idx = selected.indexOf(entry.id);
                    const isSelected = idx !== -1;
                    return (
                      <motion.button
                        key={entry.id}
                        custom={i}
                        variants={fadeUp}
                        initial="hidden"
                        animate="visible"
                        onClick={() => toggle(entry.id)}
                        className="w-full flex items-center justify-between px-3.5 py-3 rounded-xl text-left transition-all border"
                        style={{
                          background: isSelected ? `${COLORS[idx]}14` : "transparent",
                          borderColor: isSelected ? COLORS[idx] : "var(--card-border)",
                        }}
                      >
                        <div className="flex items-center gap-2.5 min-w-0">
                          {isSelected
                            ? <CheckCircle2 className="w-4 h-4 shrink-0" style={{ color: COLORS[idx] }} />
                            : <Circle className="w-4 h-4 shrink-0 text-[var(--muted)]" />}
                          <div className="min-w-0">
                            <p className="text-sm font-semibold text-[var(--foreground)] truncate">
                              {entry.startup_name}
                            </p>
                            <p className="text-xs text-[var(--muted)]">
                              {new Date(entry.timestamp).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                        <span className="text-base font-bold tabular-nums shrink-0 ml-2"
                          style={{ color: scoreColor(entry.evaluation.overall_score) }}>
                          {entry.evaluation.overall_score.toFixed(1)}
                        </span>
                      </motion.button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* ── Right: Comparison ──────────────────────────── */}
            <div className="lg:col-span-2 space-y-6">
              {picks.length < 2 ? (
                <div className="glass-card rounded-2xl p-14 text-center">
                  <GitCompare className="w-10 h-10 text-[var(--muted)] mx-auto mb-3" />
                  <p className="text-[var(--muted)]">Select at least 2 pitches to compare</p>
                </div>
              ) : (
                <>
                  {/* Radar overlay */}
                  <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                    className="glass-card rounded-2xl p-5">
                    <h3 className="text-sm font-bold text-[var(--foreground)] mb-4">Dimension Comparison</h3>
                    <ResponsiveContainer width="100%" height={320}>
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="var(--card-border)" />
                        <PolarAngleAxis dataKey="dim"
                          tick={{ fill: "var(--muted)", fontSize: 10, fontFamily: "inherit" }} />
                        <PolarRadiusAxis angle={90} domain={[0, 10]} tickCount={6}
                          tick={{ fill: "var(--muted)", fontSize: 9 }} stroke="transparent" />
                        <Tooltip contentStyle={{
                          backgroundColor: "var(--bg-secondary)",
                          border: "1px solid var(--card-border)",
                          borderRadius: "10px", fontSize: "12px",
                          color: "var(--foreground)",
                          boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
                        }} />
                        <Legend wrapperStyle={{ fontSize: "12px", color: "var(--muted)" }} />
                        {picks.map((e, i) => (
                          <Radar key={e.id} name={e.startup_name} dataKey={e.startup_name}
                            stroke={COLORS[i]} fill={COLORS[i]} fillOpacity={0.1} strokeWidth={2} />
                        ))}
                      </RadarChart>
                    </ResponsiveContainer>
                  </motion.div>

                  {/* Score table */}
                  <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="glass-card rounded-2xl p-5 overflow-x-auto">
                    <h3 className="text-sm font-bold text-[var(--foreground)] mb-4">Score Breakdown</h3>
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-[var(--card-border)]">
                          <th className="text-left py-2.5 px-2 text-[var(--muted)] font-medium text-xs uppercase tracking-wide">
                            Dimension
                          </th>
                          {picks.map((e, i) => (
                            <th key={e.id} className="text-center py-2.5 px-2 font-bold text-sm"
                              style={{ color: COLORS[i] }}>
                              {e.startup_name}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {dimNames.map((name, di) => (
                          <tr key={name} className="border-b border-[var(--card-border)]">
                            <td className="py-3 px-2 text-[var(--foreground)] capitalize text-sm">
                              {name.replace(/_/g, " ")}
                            </td>
                            {picks.map((e) => {
                              const score = e.evaluation.dimensions[di]?.score ?? 0;
                              const allScores = picks.map(p => p.evaluation.dimensions[di]?.score ?? 0);
                              const isBest = picks.length > 1 && score === Math.max(...allScores);
                              return (
                                <td key={e.id} className="text-center py-3 px-2 font-semibold tabular-nums"
                                  style={{
                                    color: scoreColor(score),
                                    fontWeight: isBest ? 800 : 600,
                                    textDecoration: isBest ? "underline" : "none",
                                  }}>
                                  {score.toFixed(1)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                        {/* Overall row */}
                        <tr className="font-bold">
                          <td className="py-3 px-2 text-[var(--foreground)] text-sm">Overall Score</td>
                          {picks.map((e) => {
                            const score = e.evaluation.overall_score;
                            const best = Math.max(...picks.map(p => p.evaluation.overall_score));
                            return (
                              <td key={e.id} className="text-center py-3 px-2 tabular-nums text-base font-black"
                                style={{
                                  color: scoreColor(score),
                                  textDecoration: picks.length > 1 && score === best ? "underline" : "none",
                                }}>
                                {score.toFixed(1)}
                              </td>
                            );
                          })}
                        </tr>
                        {/* Recommendation row */}
                        <tr>
                          <td className="py-3 px-2 text-[var(--foreground)] text-sm font-semibold">
                            Recommendation
                          </td>
                          {picks.map((e) => (
                            <td key={e.id} className="text-center py-3 px-2 text-xs font-bold text-[var(--foreground)]">
                              {e.evaluation.investment_recommendation}
                            </td>
                          ))}
                        </tr>
                      </tbody>
                    </table>
                  </motion.div>

                  {/* Variance disclaimer */}
                  <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-card rounded-2xl p-4 flex items-start gap-3"
                    style={{ borderColor: "var(--card-border)" }}>
                    <span className="text-[var(--muted)] text-lg mt-0.5">&#9432;</span>
                    <p className="text-xs text-[var(--muted)] leading-relaxed">
                      Score variations between runs are normal. PitchPal uses live web research, so different
                      data sources may be found each time, leading to slightly different assessments.
                      Differences of 1-2 points reflect real-time research variability.
                    </p>
                  </motion.div>
                </>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
