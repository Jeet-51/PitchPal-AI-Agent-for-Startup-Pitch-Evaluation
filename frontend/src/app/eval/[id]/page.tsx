"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import Link from "next/link";
import {
    Clock, Eye, Rocket, AlertTriangle,
    CheckCircle2, TrendingUp, Zap,
} from "lucide-react";
import { SharedEvaluation } from "@/types";
import EvaluationResults from "@/components/EvaluationResults";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function formatDate(ts: number) {
    return new Date(ts * 1000).toLocaleDateString("en-US", {
        month: "short", day: "numeric", year: "numeric",
    });
}

function formatExpiry(ts: number) {
    const days = Math.ceil((ts * 1000 - Date.now()) / 86400000);
    return days > 0 ? `${days} day${days !== 1 ? "s" : ""}` : "Expired";
}

const REC_COLOR: Record<string, string> = {
    "Strong Buy": "#10b981",
    "Buy": "#34d399",
    "Hold": "#f59e0b",
    "Pass": "#f87171",
    "Strong Pass": "#ef4444",
};

export default function SharedEvalPage() {
    const params = useParams();
    const shareId = params?.id as string;

    const [data, setData] = useState<SharedEvaluation | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!shareId) return;
        fetch(`${API_BASE}/eval/${shareId}`)
            .then((r) => {
                if (!r.ok) throw new Error("not_found");
                return r.json();
            })
            .then((d) => {
                setData(d);
                setLoading(false);
            })
            .catch(() => {
                setError("This evaluation link has expired or doesn't exist.");
                setLoading(false);
            });
    }, [shareId]);

    // ── Loading ───────────────────────────────────────────────
    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center"
                style={{ background: "var(--background)" }}>
                <div className="text-center space-y-4">
                    <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                    >
                        <Zap className="w-10 h-10 mx-auto" style={{ color: "var(--primary)" }} />
                    </motion.div>
                    <p className="text-[var(--muted)] text-sm">Loading shared evaluation…</p>
                </div>
            </div>
        );
    }

    // ── Error / Not found ─────────────────────────────────────
    if (error || !data) {
        return (
            <div className="min-h-screen flex items-center justify-center px-4"
                style={{ background: "var(--background)" }}>
                <div className="text-center space-y-5 max-w-md">
                    <div className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center"
                        style={{ background: "rgba(248,113,113,0.12)", border: "1px solid rgba(248,113,113,0.25)" }}>
                        <AlertTriangle className="w-8 h-8 text-red-400" />
                    </div>
                    <h1 className="text-2xl font-bold text-[var(--foreground)]">Link Expired</h1>
                    <p className="text-[var(--muted)] text-sm leading-relaxed">
                        {error || "This shared evaluation no longer exists. Shared links expire after 7 days."}
                    </p>
                    <Link href="/evaluate"
                        className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-semibold text-white"
                        style={{ background: "linear-gradient(135deg, var(--primary), var(--accent))" }}>
                        <Rocket className="w-4 h-4" />
                        Evaluate your own pitch
                    </Link>
                </div>
            </div>
        );
    }

    const { evaluation, startup_name, role, processing_time, llm_provider, created_at, expires_at, views } = data;
    const rec = evaluation.investment_recommendation;
    const recColor = REC_COLOR[rec] || "#6366f1";
    const overallPct = Math.round((evaluation.overall_score / 10) * 100);

    return (
        <div className="min-h-screen" style={{ background: "var(--background)" }}>

            {/* ── Shared banner ────────────────────────────────────── */}
            <div className="border-b py-2 px-4 flex items-center justify-between text-xs"
                style={{ background: "var(--hover-bg)", borderColor: "var(--card-border)" }}>
                <div className="flex items-center gap-3 text-[var(--muted)]">
                    <span className="flex items-center gap-1">
                        <Clock className="w-3.5 h-3.5" />
                        Shared {formatDate(created_at)} · expires in {formatExpiry(expires_at)}
                    </span>
                    <span className="hidden sm:flex items-center gap-1">
                        <Eye className="w-3.5 h-3.5" />
                        {views} view{views !== 1 ? "s" : ""}
                    </span>
                </div>
                <Link href="/evaluate"
                    className="flex items-center gap-1.5 font-semibold transition-colors hover:opacity-80"
                    style={{ color: "var(--primary)" }}>
                    <Rocket className="w-3.5 h-3.5" />
                    Try PitchPal
                </Link>
            </div>

            <main className="max-w-5xl mx-auto px-4 sm:px-6 py-10 space-y-8">

                {/* ── Header ──────────────────────────────────────────── */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.45 }}
                    className="glass-card rounded-2xl p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5"
                >
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-2xl flex items-center justify-center shrink-0"
                            style={{ background: "linear-gradient(135deg, var(--primary), var(--accent))" }}>
                            <Rocket className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-xl font-black text-[var(--foreground)]">{startup_name}</h1>
                            <p className="text-sm text-[var(--muted)] capitalize">
                                {role === "investor" ? "Investor Analysis" : "Startup Evaluation"} · powered by{" "}
                                <span className="gradient-text-static font-semibold">PitchPal</span>
                            </p>
                        </div>
                    </div>

                    {/* Score + recommendation */}
                    <div className="flex items-center gap-4">
                        <div className="text-center">
                            <div className="relative w-16 h-16">
                                <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                                    <circle cx="18" cy="18" r="15" fill="none" strokeWidth="2.5"
                                        stroke="var(--card-border)" />
                                    <circle cx="18" cy="18" r="15" fill="none" strokeWidth="2.5"
                                        stroke={recColor}
                                        strokeDasharray={`${overallPct * 0.94} 100`}
                                        strokeLinecap="round" />
                                </svg>
                                <span className="absolute inset-0 flex items-center justify-center
                  text-lg font-black" style={{ color: recColor }}>
                                    {evaluation.overall_score.toFixed(1)}
                                </span>
                            </div>
                        </div>
                        <div className="px-3 py-1.5 rounded-xl text-sm font-bold"
                            style={{ background: `${recColor}18`, color: recColor, border: `1px solid ${recColor}35` }}>
                            {rec}
                        </div>
                    </div>
                </motion.div>

                {/* ── Meta row ─────────────────────────────────────────── */}
                <div className="flex flex-wrap gap-3">
                    {[
                        { icon: Zap, label: `${processing_time.toFixed(1)}s evaluation` },
                        { icon: CheckCircle2, label: `${evaluation.dimensions.length} dimensions scored` },
                        { icon: TrendingUp, label: llm_provider },
                    ].map(({ icon: Icon, label }) => (
                        <div key={label}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium"
                            style={{ background: "var(--hover-bg)", border: "1px solid var(--card-border)", color: "var(--muted)" }}>
                            <Icon className="w-3.5 h-3.5" />
                            {label}
                        </div>
                    ))}
                </div>

                {/* ── Full evaluation results ──────────────────────────── */}
                <EvaluationResults
                    evaluation={evaluation}
                    processingTime={processing_time}
                    llmProvider={llm_provider}
                    role={role}
                    readOnly
                />

                {/* ── CTA ──────────────────────────────────────────────── */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.4 }}
                    className="rounded-2xl p-8 text-center space-y-4"
                    style={{
                        background: "linear-gradient(135deg, var(--primary)12, var(--accent)08)",
                        border: "1px solid var(--card-border)",
                    }}
                >
                    <p className="font-semibold text-[var(--foreground)]">
                        Want to evaluate your own startup?
                    </p>
                    <p className="text-sm text-[var(--muted)]">
                        PitchPal runs a live ReAct agent with real web research — not just a prompt.
                    </p>
                    <Link href="/evaluate"
                        className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold text-white"
                        style={{ background: "linear-gradient(135deg, var(--primary), var(--accent))" }}>
                        <Rocket className="w-4 h-4" />
                        Evaluate My Pitch
                    </Link>
                </motion.div>

            </main>
        </div>
    );
}
