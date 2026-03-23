"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import {
  Rocket, TrendingUp, Shield, Lock, ArrowRight,
  Brain, Search, BarChart3, Zap, Sun, Moon, Sparkles, XCircle, CheckCircle2,
} from "lucide-react";
import { setRole, setInvestorToken } from "@/lib/auth";
import { verifyInvestorCode } from "@/lib/api";
import { useTheme } from "@/components/ThemeProvider";

/* ── Animation variants ──────────────────────────────────── */
const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.1 } },
};
const item = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.55 } },
} as any;
const cardHover = {
  rest: { y: 0, scale: 1 },
  hover: { y: -8, scale: 1.01, transition: { type: "spring", stiffness: 380, damping: 22 } },
} as any;

/* ── Floating particles config ─────────────────────────── */
const PARTICLES = Array.from({ length: 20 }, (_, i) => ({
  id: i,
  size: 2 + Math.random() * 4,
  x: Math.random() * 100,
  y: Math.random() * 100,
  duration: 15 + Math.random() * 25,
  delay: Math.random() * 10,
  opacity: 0.1 + Math.random() * 0.25,
}));

const features = [
  { icon: Brain, label: "ReAct Agent", cls: "from-violet-500 to-purple-600" },
  { icon: Search, label: "Live Research", cls: "from-blue-500   to-cyan-500" },
  { icon: BarChart3, label: "5–7 Dimensions", cls: "from-emerald-500 to-teal-500" },
  { icon: Zap, label: "Real-Time Stream", cls: "from-amber-500  to-orange-500" },
];

export default function LandingPage() {
  const router = useRouter();
  const { theme, toggleTheme } = useTheme();

  const [showCode, setShowCode] = useState(false);
  const [code, setCode] = useState("");
  const [codeError, setCodeError] = useState("");
  const [verifying, setVerifying] = useState(false);

  /* Startup — no gate */
  const goStartup = () => { setRole("startup"); router.push("/evaluate"); };

  /* Investor — needs access code */
  const openInvestor = () => { setShowCode(true); setCodeError(""); };

  const submitCode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!code.trim()) return;
    setVerifying(true); setCodeError("");
    try {
      const res = await verifyInvestorCode(code.trim());
      if (res.success && res.token && res.expires_in) {
        setInvestorToken(res.token, res.expires_in);
        setRole("investor");
        router.push("/evaluate");
      } else {
        setCodeError(res.error || "Invalid access code");
      }
    } catch {
      setCodeError("Backend not reachable. Make sure the server is running.");
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--background)] relative overflow-hidden">

      {/* ── Animated mesh gradient background ────────────── */}
      <div className="mesh-gradient" />

      {/* ── Floating orb background ──────────────────────── */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="orb orb-1 absolute w-[640px] h-[640px] -top-60 -right-40" />
        <div className="orb orb-2 absolute w-[540px] h-[540px] -bottom-40 -left-40" />
        <div className="orb orb-3 absolute w-[420px] h-[420px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
      </div>

      {/* ── Subtle dot-grid ──────────────────────────────── */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `radial-gradient(circle, var(--card-border) 1px, transparent 1px)`,
          backgroundSize: "32px 32px",
          opacity: 0.35,
        }}
      />

      {/* ── Floating particles ─────────────────────────────── */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {PARTICLES.map((p) => (
          <motion.div
            key={p.id}
            className="absolute rounded-full"
            style={{
              width: p.size,
              height: p.size,
              left: `${p.x}%`,
              top: `${p.y}%`,
              background: "var(--primary)",
              opacity: p.opacity,
            }}
            animate={{
              y: [0, -80, -30, -120, 0],
              x: [0, 30, -20, 15, 0],
              scale: [1, 1.3, 0.8, 1.1, 1],
            }}
            transition={{
              duration: p.duration,
              repeat: Infinity,
              delay: p.delay,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>

      {/* ── Theme toggle ─────────────────────────────────── */}
      <motion.button
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        onClick={toggleTheme}
        className="absolute top-5 right-5 z-20 p-2.5 glass rounded-xl hover:border-[var(--primary)] text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
        title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      >
        {theme === "dark" ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
      </motion.button>

      {/* ── Main content ─────────────────────────────────── */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="visible"
        className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 py-20"
      >
        {/* Logo */}
        <motion.div variants={item} className="mb-10 text-center">
          <motion.div
            whileHover={{ scale: 1.08, rotate: -6 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-6"
            style={{
              background: "linear-gradient(135deg, var(--primary), var(--accent))",
              boxShadow: "0 0 48px var(--primary-glow), 0 12px 32px rgba(0,0,0,0.18)",
            }}
          >
            <Rocket className="w-10 h-10 text-white" />
          </motion.div>

          <h1 className="text-7xl sm:text-8xl font-black tracking-tight mb-4 leading-none">
            <span className="gradient-text">PitchPal</span>
          </h1>
          <p className="text-lg text-[var(--muted)] max-w-md mx-auto leading-relaxed">
            AI-powered pitch evaluation with a real{" "}
            <span className="text-[var(--primary)] font-semibold">ReAct agent</span> and live web research
          </p>
        </motion.div>

        {/* Feature pills */}
        <motion.div variants={item} className="flex flex-wrap justify-center gap-2.5 mb-12">
          {features.map((f) => (
            <motion.div
              key={f.label}
              whileHover={{ scale: 1.06, y: -2 }}
              className="flex items-center gap-2 px-3.5 py-2 glass rounded-full text-xs font-medium text-[var(--foreground)] cursor-default"
            >
              <span className={`w-5 h-5 rounded-full bg-gradient-to-br ${f.cls} flex items-center justify-center shrink-0`}>
                <f.icon className="w-3 h-3 text-white" />
              </span>
              {f.label}
            </motion.div>
          ))}
        </motion.div>

        {/* ── Option 2: Secondary "Why?" CTA ───────────────── */}
        <motion.div variants={item} className="flex items-center justify-center gap-2 mb-8">
          <span className="text-xs text-[var(--muted)]">Not just another AI prompt —</span>
          <Link
            href="/why"
            className="flex items-center gap-1 text-xs font-semibold transition-all"
            style={{ color: "var(--accent-primary)" }}
          >
            <Sparkles className="w-3.5 h-3.5" />
            See how PitchPal outperforms Generic AI Prompts
            <ArrowRight className="w-3 h-3" />
          </Link>
        </motion.div>

        {/* Role headline */}
        <motion.div variants={item} className="text-center mb-6">
          <h2 className="text-xl font-semibold text-[var(--foreground)]">Choose your perspective</h2>
          <p className="text-sm text-[var(--muted)] mt-1">
            Each role unlocks a tailored evaluation lens
          </p>
        </motion.div>

        {/* ── Role cards ───────────────────────────────────── */}
        <motion.div variants={item} className="grid grid-cols-1 md:grid-cols-2 gap-5 max-w-3xl w-full">

          {/* Startup card */}
          <motion.button
            variants={cardHover}
            initial="rest"
            whileHover="hover"
            whileTap={{ scale: 0.98 }}
            onClick={goStartup}
            className="group relative glass-card rounded-3xl p-8 text-left overflow-hidden"
          >
            {/* Top accent line */}
            <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-cyan-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            {/* Inner glow */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/[0.05] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-3xl pointer-events-none" />

            <div className="relative">
              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center mb-5"
                style={{ background: "linear-gradient(135deg,#3b82f6,#6366f1)", boxShadow: "0 8px 24px rgba(59,130,246,0.35)" }}
              >
                <TrendingUp className="w-7 h-7 text-white" />
              </div>

              <h3 className="text-xl font-bold text-[var(--foreground)] mb-2">Startup Founder</h3>
              <p className="text-sm text-[var(--muted)] mb-5 leading-relaxed">
                Actionable feedback across 5 key dimensions. Understand your strengths and where to sharpen your pitch.
              </p>

              <div className="space-y-1.5 mb-6">
                {["Problem Clarity", "Market Opportunity", "Business Model", "Competitive Advantage", "Team Strength"].map((d) => (
                  <div key={d} className="flex items-center gap-2 text-xs text-[var(--muted)]">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />
                    {d}
                  </div>
                ))}
              </div>

              <div className="flex items-center gap-2 text-sm font-semibold text-blue-500 group-hover:gap-3 transition-all duration-200">
                Get Started <ArrowRight className="w-4 h-4" />
              </div>
            </div>
          </motion.button>

          {/* Investor card */}
          <div className="relative">
            <motion.button
              variants={cardHover}
              initial="rest"
              whileHover={!showCode ? "hover" : "rest"}
              whileTap={!showCode ? { scale: 0.98 } : {}}
              onClick={openInvestor}
              className="group relative glass-card rounded-3xl p-8 text-left overflow-hidden w-full"
            >
              <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-purple-500 via-violet-500 to-pink-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500/[0.05] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-3xl pointer-events-none" />

              <div className="relative">
                <div className="flex items-center gap-3 mb-5">
                  <div
                    className="w-14 h-14 rounded-2xl flex items-center justify-center"
                    style={{ background: "linear-gradient(135deg,#8b5cf6,#7c3aed)", boxShadow: "0 8px 24px rgba(139,92,246,0.35)" }}
                  >
                    <Shield className="w-7 h-7 text-white" />
                  </div>
                  <span className="flex items-center gap-1.5 px-2.5 py-1 glass rounded-full border border-purple-500/30 text-xs font-medium text-purple-400">
                    <Lock className="w-3 h-3" /> Access Code
                  </span>
                </div>

                <h3 className="text-xl font-bold text-[var(--foreground)] mb-2">Investor / VC</h3>
                <p className="text-sm text-[var(--muted)] mb-5 leading-relaxed">
                  Investment-grade analysis: unit economics, exit potential, scalability, and risk across 7 dimensions.
                </p>

                <div className="space-y-1.5 mb-6">
                  {["Market Opportunity", "Revenue & Unit Economics", "Scalability", "Competitive Moat", "Team & Execution", "Risk Assessment", "Exit Potential"].map((d) => (
                    <div key={d} className="flex items-center gap-2 text-xs text-[var(--muted)]">
                      <div className="w-1.5 h-1.5 rounded-full bg-purple-400 shrink-0" />
                      {d}
                    </div>
                  ))}
                </div>

                <div className="flex items-center gap-2 text-sm font-semibold text-purple-400 group-hover:gap-3 transition-all duration-200">
                  Enter Access Code <ArrowRight className="w-4 h-4" />
                </div>
                <p className="text-xs text-[var(--muted)] mt-2 opacity-70">
                  Contact <a href="https://www.linkedin.com/in/pateljeet22/" target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:text-purple-300 underline underline-offset-2" onClick={(e) => e.stopPropagation()}>Jeet Patel</a> to get the access code
                </p>
              </div>
            </motion.button>

            {/* Access code overlay */}
            <AnimatePresence>
              {showCode && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.97 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.97 }}
                  transition={{ duration: 0.2 }}
                  className="absolute inset-0 rounded-3xl flex items-center justify-center p-6 z-10 border border-purple-500/40"
                  style={{ background: "var(--card-bg)", backdropFilter: "blur(28px)", WebkitBackdropFilter: "blur(28px)" }}
                >
                  <form onSubmit={submitCode} className="w-full max-w-xs space-y-4">
                    <div className="text-center">
                      <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-3"
                        style={{ background: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.3)" }}>
                        <Lock className="w-5 h-5 text-purple-400" />
                      </div>
                      <h4 className="font-bold text-[var(--foreground)]">Investor Access</h4>
                      <p className="text-xs text-[var(--muted)] mt-1">Enter the access code to unlock investor-grade analysis</p>
                    </div>

                    <input
                      type="password"
                      value={code}
                      onChange={(e) => setCode(e.target.value)}
                      placeholder="Enter access code"
                      className="w-full px-4 py-3 rounded-xl glass-input text-sm"
                      autoFocus
                    />

                    {codeError && <p className="text-xs text-[var(--danger)] text-center">{codeError}</p>}

                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => { setShowCode(false); setCode(""); setCodeError(""); }}
                        className="flex-1 px-4 py-2.5 text-sm glass rounded-xl text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        disabled={verifying || !code.trim()}
                        className="flex-1 px-4 py-2.5 text-sm text-white font-medium rounded-xl disabled:opacity-50 transition-opacity"
                        style={{ background: "linear-gradient(135deg,#8b5cf6,#7c3aed)" }}
                      >
                        {verifying ? "Verifying…" : "Verify"}
                      </button>
                    </div>
                  </form>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* ── Option 1: Teaser comparison section ─────────── */}
        <motion.div variants={item} className="mt-16 w-full max-w-3xl">
          {/* Headline */}
          <div className="text-center mb-6 space-y-1">
            <p className="text-xs uppercase tracking-widest font-semibold" style={{ color: "var(--accent-primary)" }}>
              Why not just use a generic AI prompt?
            </p>
            <h2 className="text-xl font-bold text-[var(--foreground)]">
              Same pitch. Completely different output.
            </h2>
          </div>

          {/* Mini side-by-side */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">

            {/* Generic AI side */}
            <div className="glass-card rounded-2xl p-5 space-y-3"
              style={{ borderColor: "var(--card-border)" }}>
              <div className="flex items-center gap-2">
                <span className="text-base">🤖</span>
                <span className="text-sm font-semibold text-[var(--muted)]">Generic AI Prompt</span>
              </div>
              <p className="text-xs text-[var(--muted)] leading-relaxed italic opacity-80">
                &quot;Your pitch looks promising. The market seems large but needs validation.
                The team appears capable. Consider your go-to-market strategy and competitive moat.&quot;
              </p>
              <div className="flex flex-wrap gap-2 pt-1">
                {["No score", "No sources", "Generic"].map(t => (
                  <span key={t} className="flex items-center gap-1 text-[10px] text-red-400">
                    <XCircle className="w-3 h-3" />{t}
                  </span>
                ))}
              </div>
            </div>

            {/* PitchPal side */}
            <div className="rounded-2xl p-5 space-y-3"
              style={{
                background: "var(--card-bg)",
                border: "1px solid var(--accent-primary)40",
                boxShadow: "0 0 24px var(--primary-glow)",
              }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4" style={{ color: "var(--accent-primary)" }} />
                  <span className="text-sm font-semibold text-[var(--foreground)]">PitchPal</span>
                </div>
                <span className="text-xs font-bold px-2 py-0.5 rounded-full"
                  style={{ background: "#10b98120", color: "#10b981" }}>8.0 / 10</span>
              </div>
              {[
                { label: "Market Opportunity", score: 8.0, color: "#10b981" },
                { label: "Team & Execution", score: 9.0, color: "#10b981" },
                { label: "Competitive Moat", score: 6.0, color: "#f59e0b" },
              ].map(dim => (
                <div key={dim.label} className="space-y-1">
                  <div className="flex justify-between text-[10px]">
                    <span className="text-[var(--muted)]">{dim.label}</span>
                    <span className="font-bold tabular-nums" style={{ color: dim.color }}>{dim.score.toFixed(1)}</span>
                  </div>
                  <div className="h-1 rounded-full overflow-hidden" style={{ background: "var(--card-border)" }}>
                    <div className="h-full rounded-full" style={{ width: `${dim.score * 10}%`, background: dim.color }} />
                  </div>
                </div>
              ))}
              <div className="flex flex-wrap gap-2 pt-1">
                {["Live research", "Source-backed", "Structured"].map(t => (
                  <span key={t} className="flex items-center gap-1 text-[10px]" style={{ color: "#10b981" }}>
                    <CheckCircle2 className="w-3 h-3" />{t}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* CTA to /why */}
          <div className="text-center mt-5">
            <Link
              href="/why"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all"
              style={{
                background: "var(--hover-bg)",
                border: "1px solid var(--card-border)",
                color: "var(--foreground)",
              }}
            >
              <Sparkles className="w-4 h-4" style={{ color: "var(--accent-primary)" }} />
              See the full comparison
              <ArrowRight className="w-4 h-4" style={{ color: "var(--accent-primary)" }} />
            </Link>
          </div>
        </motion.div>

        {/* Footer */}
        <motion.footer variants={item} className="mt-12 text-center">
          <p className="text-sm text-[var(--muted)]">
            Built by{" "}
            <a href="https://github.com/Jeet-51" target="_blank" rel="noopener noreferrer"
              className="text-[var(--primary)] hover:underline font-medium">
              Jeet Patel
            </a>{" "}
            · Powered by ReAct Agent · Gemini · Tavily Search
          </p>
        </motion.footer>
      </motion.div>
    </div>
  );
}
