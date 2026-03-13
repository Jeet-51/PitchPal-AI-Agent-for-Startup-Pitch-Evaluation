"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import Header from "@/components/Header";
import {
    CheckCircle2, XCircle, Search, Brain, BarChart3,
    Globe, Zap, Database, TrendingUp, Shield, Layers,
    ArrowRight, Sparkles,
} from "lucide-react";

// ── Animation helpers ───────────────────────────────────────
const fadeUp = (delay = 0) => ({
    initial: { opacity: 0, y: 24 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true },
    transition: { duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] },
});

// ── Static example evaluation data ─────────────────────────
const GENERIC_AI_RESPONSE = `Your pitch looks promising! Here are some thoughts:

The problem you're solving seems important and the market could be large. However, you should validate whether customers actually want this solution.

Your business model makes sense, though you'll need to figure out unit economics. The team appears capable but could benefit from more industry experience.

Consider strengthening your competitive moat and thinking about how you'll differentiate from existing players. Make sure you have a clear go-to-market strategy.

Overall, this has potential but needs more development before approaching investors.`;

const PITCHPAL_DIMENSIONS = [
    { name: "Market Opportunity", score: 8.0, color: "#10b981", reasoning: "Last-mile pharma delivery in SSA has $2.3B TAM growing at 18% CAGR. Mordor Intelligence confirms 400M+ people lack reliable access.", source: "mordorintelligence.com" },
    { name: "Competitive Moat", score: 6.0, color: "#f59e0b", reasoning: "Cipla Medpro, Sygen, Morison are funded competitors. PharmaBridge needs clearer IP or partnership-based differentiation.", source: "tracxn.com" },
    { name: "Team & Execution", score: 9.0, color: "#10b981", reasoning: "Founding team has 15+ years combined experience in African healthcare logistics. Government pilot signals credibility.", source: "crunchbase.com" },
    { name: "Revenue & Unit Economics", score: 6.0, color: "#f59e0b", reasoning: "B2B healthcare logistics in SSA sees $8–$14 gross margin per delivery. LTV/CAC benchmarks needed for investor confidence.", source: "mckinsey.com" },
];

// ── Feature comparison data ─────────────────────────────────
const FEATURES = [
    { label: "Live web research per evaluation", pitchpal: true, genericAi: false },
    { label: "Source-backed reasoning", pitchpal: true, genericAi: false },
    { label: "Structured dimension scoring (0–10)", pitchpal: true, genericAi: false },
    { label: "Startup vs. Investor lens", pitchpal: true, genericAi: false },
    { label: "Real competitor data from Crunchbase / Tracxn", pitchpal: true, genericAi: false },
    { label: "Current market size from live research reports", pitchpal: true, genericAi: false },
    { label: "Consistent, deterministic scoring", pitchpal: true, genericAi: false },
    { label: "Multi-step reasoning (ReAct agent)", pitchpal: true, genericAi: false },
    { label: "Evaluation history & side-by-side comparison", pitchpal: true, genericAi: false },
];

// ── ReAct agent steps ───────────────────────────────────────
const REACT_STEPS = [
    { icon: Brain, label: "Thought", desc: "I need to research the SSA pharmaceutical delivery market size.", color: "#6366f1" },
    { icon: Search, label: "Action", desc: 'market_research("last-mile pharma delivery Sub-Saharan Africa market size 2024")', color: "#10b981" },
    { icon: Globe, label: "Observation", desc: "Mordor Intelligence: Market valued at $2.3B, 18.2% CAGR through 2028. 400M+ people underserved.", color: "#f59e0b" },
    { icon: Brain, label: "Thought", desc: "Now I need to find real competitors and their funding status.", color: "#6366f1" },
    { icon: Search, label: "Action", desc: 'competitor_analysis("pharmaceutical last-mile delivery Africa startups funded")', color: "#10b981" },
    { icon: Globe, label: "Observation", desc: "Tracxn: Cipla Medpro (Series B), Sygen ($12M raised), Morison Pharmaceuticals — all operating in SSA.", color: "#f59e0b" },
    { icon: BarChart3, label: "Final Evaluation", desc: "Synthesizing all research into 7 structured dimensions with evidence-backed scores.", color: "#ec4899" },
];

// ── Stats ───────────────────────────────────────────────────
const STATS = [
    { value: "~20s", label: "Per evaluation", icon: Zap },
    { value: "7", label: "Scored dimensions", icon: Layers },
    { value: "4+", label: "Live searches per pitch", icon: Search },
    { value: "0°", label: "Randomness (temp=0)", icon: Shield },
];

export default function WhyPage() {
    return (
        <div className="min-h-screen" style={{ background: "var(--background)" }}>
            <Header />

            <main className="max-w-6xl mx-auto px-4 sm:px-6 py-16 space-y-28">

                {/* ── Hero ─────────────────────────────────────────── */}
                <section className="text-center space-y-6">
                    <motion.div {...fadeUp(0)}
                        className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold border mb-2"
                        style={{
                            background: "var(--accent-primary)15",
                            borderColor: "var(--accent-primary)40",
                            color: "var(--accent-primary)",
                        }}
                    >
                        <Sparkles className="w-3.5 h-3.5" />
                        PitchPal vs Generic AI
                    </motion.div>

                    <motion.h1 {...fadeUp(0.05)}
                        className="text-4xl sm:text-5xl font-black text-[var(--foreground)] leading-tight"
                    >
                        Generic AI Prompts guess.
                        <br />
                        <span className="gradient-text-static">PitchPal researches.</span>
                    </motion.h1>

                    <motion.p {...fadeUp(0.1)}
                        className="max-w-2xl mx-auto text-lg text-[var(--muted)] leading-relaxed"
                    >
                        Any AI can generate generic pitch feedback. PitchPal runs a live{" "}
                        <span className="text-[var(--foreground)] font-semibold">ReAct agent</span> that searches
                        the web in real-time, pulls actual market data, competitor funding, and industry benchmarks
                        — then scores your pitch across structured dimensions with source-backed reasoning.
                    </motion.p>

                    <motion.div {...fadeUp(0.15)} className="flex items-center justify-center gap-4 pt-2">
                        <Link
                            href="/evaluate"
                            className="btn-primary px-6 py-2.5 rounded-xl font-semibold text-sm flex items-center gap-2"
                        >
                            Try PitchPal Free
                            <ArrowRight className="w-4 h-4" />
                        </Link>
                    </motion.div>
                </section>

                {/* ── Side-by-side comparison ───────────────────────── */}
                <section className="space-y-6">
                    <motion.div {...fadeUp(0)} className="text-center space-y-2">
                        <h2 className="text-2xl font-bold text-[var(--foreground)]">Same pitch. Completely different output.</h2>
                        <p className="text-[var(--muted)] text-sm">Evaluated pitch: <em>PharmaBridge — last-mile pharmaceutical delivery, Sub-Saharan Africa</em></p>
                    </motion.div>

                    <div className="grid md:grid-cols-2 gap-5">

                        {/* Generic AI card */}
                        <motion.div {...fadeUp(0.05)} className="rounded-2xl border overflow-hidden"
                            style={{ borderColor: "var(--card-border)", background: "var(--card-bg)" }}>
                            <div className="flex items-center gap-3 px-5 py-4 border-b"
                                style={{ borderColor: "var(--card-border)", background: "var(--hover-bg)" }}>
                                <div className="w-8 h-8 rounded-xl bg-gray-500/20 flex items-center justify-center text-base">🤖</div>
                                <div>
                                    <p className="font-semibold text-sm text-[var(--foreground)]">Generic AI Prompt</p>
                                    <p className="text-xs text-[var(--muted)]">No research · No sources · No scores</p>
                                </div>
                            </div>
                            <div className="p-5 space-y-4">
                                <div className="rounded-xl p-4"
                                    style={{ background: "var(--hover-bg)", border: "1px solid var(--card-border)" }}>
                                    <p className="text-xs text-[var(--muted)] mb-2 uppercase tracking-wide font-semibold">Response</p>
                                    <p className="text-sm text-[var(--foreground)] leading-relaxed whitespace-pre-line opacity-80">
                                        {GENERIC_AI_RESPONSE}
                                    </p>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    {["No score", "No sources", "No structure", "Generic"].map(tag => (
                                        <div key={tag} className="flex items-center gap-2 text-xs text-[var(--muted)]">
                                            <XCircle className="w-4 h-4 text-red-500 shrink-0" />
                                            {tag}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </motion.div>

                        {/* PitchPal card */}
                        <motion.div {...fadeUp(0.1)} className="rounded-2xl border overflow-hidden"
                            style={{
                                borderColor: "var(--accent-primary)50",
                                background: "var(--card-bg)",
                                boxShadow: "0 0 40px var(--primary-glow)",
                            }}>
                            <div className="flex items-center justify-between px-5 py-4 border-b"
                                style={{
                                    borderColor: "var(--accent-primary)30",
                                    background: "linear-gradient(135deg, var(--primary)10, var(--accent)08)",
                                }}>
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-xl flex items-center justify-center"
                                        style={{ background: "linear-gradient(135deg, var(--primary), var(--accent))" }}>
                                        <Zap className="w-4 h-4 text-white" />
                                    </div>
                                    <div>
                                        <p className="font-semibold text-sm text-[var(--foreground)]">PitchPal ReAct Agent</p>
                                        <p className="text-xs text-[var(--muted)]">Live research · Real data · Structured scoring</p>
                                    </div>
                                </div>
                                <div className="px-2.5 py-1 rounded-full text-xs font-bold"
                                    style={{ background: "#10b98120", color: "#10b981" }}>
                                    8.0 / 10
                                </div>
                            </div>
                            <div className="p-5 space-y-3">
                                {PITCHPAL_DIMENSIONS.map((dim, i) => (
                                    <motion.div
                                        key={dim.name}
                                        initial={{ opacity: 0, x: 16 }}
                                        whileInView={{ opacity: 1, x: 0 }}
                                        viewport={{ once: true }}
                                        transition={{ delay: 0.1 + i * 0.08, duration: 0.4 }}
                                        className="rounded-xl p-3.5 space-y-2"
                                        style={{ background: "var(--hover-bg)", border: "1px solid var(--card-border)" }}
                                    >
                                        <div className="flex items-center justify-between">
                                            <span className="text-xs font-semibold text-[var(--foreground)]">{dim.name}</span>
                                            <span className="text-sm font-black tabular-nums" style={{ color: dim.color }}>
                                                {dim.score.toFixed(1)}
                                            </span>
                                        </div>
                                        {/* Score bar */}
                                        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--card-border)" }}>
                                            <motion.div
                                                initial={{ width: 0 }}
                                                whileInView={{ width: `${dim.score * 10}%` }}
                                                viewport={{ once: true }}
                                                transition={{ delay: 0.2 + i * 0.08, duration: 0.6 }}
                                                className="h-full rounded-full"
                                                style={{ background: dim.color }}
                                            />
                                        </div>
                                        <p className="text-xs text-[var(--muted)] leading-relaxed">{dim.reasoning}</p>
                                        <p className="text-xs font-medium" style={{ color: "var(--accent-primary)" }}>
                                            📡 Source: {dim.source}
                                        </p>
                                    </motion.div>
                                ))}
                            </div>
                        </motion.div>

                    </div>
                </section>

                {/* ── Feature comparison table ─────────────────────── */}
                <section className="space-y-6">
                    <motion.h2 {...fadeUp(0)} className="text-2xl font-bold text-[var(--foreground)] text-center">
                        Feature by Feature
                    </motion.h2>

                    <motion.div {...fadeUp(0.05)}
                        className="rounded-2xl overflow-hidden border"
                        style={{ borderColor: "var(--card-border)" }}>
                        {/* Header row */}
                        <div className="grid grid-cols-3 px-6 py-3 border-b text-xs font-semibold uppercase tracking-wide"
                            style={{ borderColor: "var(--card-border)", background: "var(--hover-bg)" }}>
                            <span className="text-[var(--muted)]">Capability</span>
                            <span className="text-center text-[var(--muted)]">Generic AI</span>
                            <span className="text-center gradient-text-static">PitchPal</span>
                        </div>
                        {FEATURES.map((f, i) => (
                            <div key={f.label}
                                className="grid grid-cols-3 items-center px-6 py-3.5 border-b text-sm"
                                style={{
                                    borderColor: "var(--card-border)",
                                    background: i % 2 === 0 ? "var(--card-bg)" : "var(--hover-bg)",
                                }}>
                                <span className="text-[var(--foreground)]">{f.label}</span>
                                <div className="flex justify-center">
                                    {f.genericAi
                                        ? <CheckCircle2 className="w-5 h-5" style={{ color: "#6b7280" }} />
                                        : <XCircle className="w-5 h-5 text-red-500/60" />}
                                </div>
                                <div className="flex justify-center">
                                    {f.pitchpal
                                        ? <CheckCircle2 className="w-5 h-5" style={{ color: "#10b981" }} />
                                        : <XCircle className="w-5 h-5 text-red-500/60" />}
                                </div>
                            </div>
                        ))}
                    </motion.div>
                </section>

                {/* ── How it works: ReAct loop ─────────────────────── */}
                <section className="space-y-8">
                    <motion.div {...fadeUp(0)} className="text-center space-y-2">
                        <h2 className="text-2xl font-bold text-[var(--foreground)]">The ReAct Agent Loop</h2>
                        <p className="text-[var(--muted)] text-sm max-w-xl mx-auto">
                            Unlike a simple prompt, PitchPal runs a multi-step reasoning loop — Thought → Action → Observation —
                            repeatedly until it has enough real data to write a rigorous evaluation.
                        </p>
                    </motion.div>

                    <div className="relative">
                        {/* Vertical line */}
                        <div className="absolute left-6 top-4 bottom-4 w-0.5 hidden sm:block"
                            style={{ background: "linear-gradient(to bottom, var(--primary), var(--accent))" }} />

                        <div className="space-y-3">
                            {REACT_STEPS.map((step, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: -20 }}
                                    whileInView={{ opacity: 1, x: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ delay: i * 0.07, duration: 0.4 }}
                                    className="flex items-start gap-4 sm:ml-12 rounded-xl p-4"
                                    style={{ background: "var(--card-bg)", border: "1px solid var(--card-border)" }}
                                >
                                    <div className="shrink-0 w-9 h-9 rounded-xl flex items-center justify-center sm:-ml-16"
                                        style={{ background: `${step.color}20`, border: `1px solid ${step.color}40` }}>
                                        <step.icon className="w-4 h-4" style={{ color: step.color }} />
                                    </div>
                                    <div className="sm:ml-4">
                                        <span className="text-xs font-bold uppercase tracking-wide" style={{ color: step.color }}>
                                            {step.label}
                                        </span>
                                        <p className="text-sm text-[var(--foreground)] mt-0.5 font-mono leading-relaxed">{step.desc}</p>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* ── Stats ────────────────────────────────────────── */}
                <section>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        {STATS.map((s, i) => (
                            <motion.div
                                key={s.label}
                                {...fadeUp(i * 0.07)}
                                className="glass-card rounded-2xl p-5 text-center space-y-2"
                            >
                                <s.icon className="w-6 h-6 mx-auto" style={{ color: "var(--accent-primary)" }} />
                                <p className="text-3xl font-black gradient-text-static">{s.value}</p>
                                <p className="text-xs text-[var(--muted)] font-medium">{s.label}</p>
                            </motion.div>
                        ))}
                    </div>
                </section>

                {/* ── CTA ──────────────────────────────────────────── */}
                <motion.section {...fadeUp(0)}
                    className="rounded-2xl p-10 text-center space-y-5"
                    style={{
                        background: "linear-gradient(135deg, var(--primary)15, var(--accent)10)",
                        border: "1px solid var(--accent-primary)30",
                    }}
                >
                    <h2 className="text-2xl sm:text-3xl font-black text-[var(--foreground)]">
                        Ready to see real evaluation in action?
                    </h2>
                    <p className="text-[var(--muted)] max-w-lg mx-auto">
                        Paste any startup pitch and watch the ReAct agent research it live — with real market data,
                        competitor intelligence, and structured scoring.
                    </p>
                    <Link
                        href="/evaluate"
                        className="btn-primary inline-flex items-center gap-2 px-8 py-3 rounded-xl font-bold text-sm"
                    >
                        <Zap className="w-4 h-4" />
                        Evaluate a Pitch Now
                    </Link>
                </motion.section>

            </main>
        </div>
    );
}
