"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import Header from "@/components/Header";
import PitchForm from "@/components/PitchForm";
import DeckUpload from "@/components/DeckUpload";
import AgentStream from "@/components/AgentStream";
import EvaluationResults from "@/components/EvaluationResults";
import { AgentStep, PitchEvaluation, WSMessage, UserRole } from "@/types";
import { getWebSocketURL } from "@/lib/api";
import { saveEvaluation } from "@/lib/storage";
import { getRole, getInvestorToken } from "@/lib/auth";
import { Brain, BarChart3, Search, Zap, Shield, Sparkles, X, ArrowRight, FileUp, Type, Clock, AlertTriangle } from "lucide-react";
import Link from "next/link";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface RateLimitStatus {
  count: number;
  limit: number;
  remaining: number;
  window_hours: number;
  retry_after_seconds: number;
  reset_at: number | null;
}

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
} as any;

export default function EvaluatePage() {
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [evaluation, setEvaluation] = useState<PitchEvaluation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [llmProvider, setLlmProvider] = useState<string | null>(null);
  const [cacheHits, setCacheHits] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [role, setRoleState] = useState<UserRole>("startup");
  const [inputMode, setInputMode] = useState<"text" | "deck">("text");
  const [rateLimitStatus, setRateLimitStatus] = useState<RateLimitStatus | null>(null);
  const [rateLimitBlocked, setRateLimitBlocked] = useState(false);
  const [retryAfter, setRetryAfter] = useState(0);

  const [showBanner, setShowBanner] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const pitchDataRef = useRef<{ name: string; text: string }>({ name: "", text: "" });
  const router = useRouter();

  // ── Countdown timer for rate limit ────────────────────────
  useEffect(() => {
    if (!rateLimitBlocked || retryAfter <= 0) return;
    const interval = setInterval(() => {
      setRetryAfter((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setRateLimitBlocked(false);
          fetchRateLimitStatus();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [rateLimitBlocked, retryAfter]);

  // ── Fetch rate limit status ───────────────────────────────
  const fetchRateLimitStatus = useCallback(async (currentRole?: UserRole) => {
    const r = currentRole || role;
    try {
      const res = await fetch(`${API_BASE}/rate-limit/status?role=${r}`);
      if (res.ok) {
        const data: RateLimitStatus = await res.json();
        setRateLimitStatus(data);
        if (data.remaining === 0 && data.retry_after_seconds > 0) {
          setRateLimitBlocked(true);
          setRetryAfter(data.retry_after_seconds);
        }
      }
    } catch {
      // non-critical — silently fail
    }
  }, [role]);

  // Show banner once per session
  useEffect(() => {
    const dismissed = sessionStorage.getItem("why-banner-dismissed");
    if (!dismissed) setShowBanner(true);
  }, []);

  const dismissBanner = () => {
    sessionStorage.setItem("why-banner-dismissed", "1");
    setShowBanner(false);
  };

  useEffect(() => {
    const r = getRole();
    if (!r) { router.push("/"); return; }
    setRoleState(r);
    fetchRateLimitStatus(r);
  }, [router]);

  const dimCount = role === "investor" ? 7 : 5;

  const handleSubmit = useCallback(
    (startupName: string, pitchText: string) => {
      setSteps([]); setEvaluation(null); setProcessingTime(null);
      setLlmProvider(null); setCacheHits(0); setError(null);
      setIsLoading(true);
      pitchDataRef.current = { name: startupName, text: pitchText };

      if (wsRef.current) wsRef.current.close();

      const ws = new WebSocket(getWebSocketURL());
      wsRef.current = ws;

      ws.onopen = () => {
        const payload: Record<string, string> = {
          startup_name: startupName,
          pitch_text: pitchText,
          role,
        };
        if (role === "investor") {
          const token = getInvestorToken();
          if (token) payload.token = token;
        }
        ws.send(JSON.stringify(payload));
      };

      ws.onmessage = (event) => {
        const msg: WSMessage = JSON.parse(event.data);
        switch (msg.type) {
          case "step":
            if (msg.step) setSteps((prev) => [...prev, msg.step!]);
            break;
          case "complete":
            if (msg.evaluation) {
              setEvaluation(msg.evaluation);
              setProcessingTime(msg.processing_time ?? null);
              setLlmProvider(msg.llm_provider ?? null);
              setCacheHits(msg.cache_hits ?? 0);
              setIsLoading(false);
              saveEvaluation({
                id: crypto.randomUUID(),
                timestamp: new Date().toISOString(),
                startup_name: pitchDataRef.current.name,
                pitch_text: pitchDataRef.current.text,
                evaluation: msg.evaluation,
                processing_time: msg.processing_time ?? 0,
                llm_provider: msg.llm_provider ?? "unknown",
                role,
              });
              setTimeout(() => {
                resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
              }, 300);
            }
            break;
          case "rate_limit": {
            const retryS = msg.retry_after_seconds || 0;
            setRateLimitBlocked(true);
            setRetryAfter(retryS);
            setError(msg.message || "Rate limit exceeded");
            setIsLoading(false);
            fetchRateLimitStatus();
            break;
          }
          case "error":
            setError(msg.message || "An error occurred");
            setIsLoading(false);
            break;
        }
      };

      ws.onerror = () => {
        setError("Could not connect to the API server. Make sure the backend is running on port 8000.");
        setIsLoading(false);
      };

      ws.onclose = () => { };
    },
    [role]
  );

  const pills = [
    { icon: Brain, label: "ReAct Agent", cls: "from-violet-500 to-purple-600" },
    { icon: Search, label: "Live Web Research", cls: "from-blue-500   to-cyan-500" },
    { icon: BarChart3, label: `${dimCount}-Dimension`, cls: "from-emerald-500 to-teal-500" },
    ...(role === "investor"
      ? [{ icon: Shield, label: "Investment Analysis", cls: "from-purple-500 to-pink-500" }]
      : [{ icon: Zap, label: "Real-Time Stream", cls: "from-amber-500  to-orange-500" }]),
  ];

  const formatRetry = (secs: number) => {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Subtle orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="orb orb-1 absolute w-[500px] h-[500px] -top-48 -right-32 opacity-60" />
        <div className="orb orb-2 absolute w-[400px] h-[400px] -bottom-32 -left-32 opacity-50" />
      </div>

      <Header />

      {/* ── Why PitchPal Banner ───────────────────────── */}
      <AnimatePresence>
        {showBanner && (
          <motion.div
            key="why-banner"
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            className="relative"
            style={{
              background: "linear-gradient(90deg, var(--primary)12, var(--accent)08, transparent)",
              borderBottom: "1px solid var(--accent-primary)25",
              borderLeft: "4px solid var(--accent-primary)",
            }}
          >
            <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <Sparkles className="w-4 h-4 shrink-0" style={{ color: "var(--accent-primary)" }} />
                <p className="text-sm text-[var(--foreground)]">
                  <span className="font-semibold">Wondering how this is different from generic AI prompts?</span>
                  <span className="text-[var(--muted)] ml-1 hidden sm:inline">
                    PitchPal runs live web research — not just a prompt.
                  </span>
                </p>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                <Link
                  href="/why"
                  onClick={dismissBanner}
                  className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all"
                  style={{
                    background: "var(--accent-primary)20",
                    color: "var(--accent-primary)",
                    border: "1px solid var(--accent-primary)40",
                  }}
                >
                  See why
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
                <button
                  onClick={dismissBanner}
                  className="p-1 rounded-lg text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--hover-bg)] transition-colors"
                  aria-label="Dismiss"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">

        {/* ── Hero (only before evaluation starts) ──────── */}
        <AnimatePresence>
          {!evaluation && steps.length === 0 && !isLoading && (
            <motion.div
              key="hero"
              variants={fadeUp}
              initial="hidden"
              animate="visible"
              exit={{ opacity: 0, y: -10, transition: { duration: 0.3 } }}
              className="text-center mb-10"
            >
              <h2 className="text-3xl sm:text-4xl font-bold text-[var(--foreground)] mb-3">
                {role === "investor" ? (
                  <>Investor-Grade{" "}<span className="gradient-text-static">Due Diligence</span></>
                ) : (
                  <>Evaluate with{" "}<span className="gradient-text-static">Real AI Research</span></>
                )}
              </h2>
              <p className="text-[var(--muted)] max-w-xl mx-auto">
                {role === "investor"
                  ? `Deep-dive analysis across ${dimCount} investment dimensions with live market research.`
                  : `Watch a ReAct AI agent research real data and evaluate your pitch across ${dimCount} dimensions — live.`}
              </p>

              {/* Feature pills */}
              <div className="flex flex-wrap justify-center gap-2.5 mt-6">
                {pills.map((p) => (
                  <motion.div
                    key={p.label}
                    whileHover={{ scale: 1.05, y: -2 }}
                    className="flex items-center gap-2 px-3.5 py-2 glass rounded-full text-xs font-medium text-[var(--foreground)]"
                  >
                    <span className={`w-5 h-5 rounded-full bg-gradient-to-br ${p.cls} flex items-center justify-center shrink-0`}>
                      <p.icon className="w-3 h-3 text-white" />
                    </span>
                    {p.label}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Rate Limit Counter Badge ──────────────────────── */}
        {rateLimitStatus && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-end mb-4"
          >
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium"
              style={{
                background: rateLimitBlocked
                  ? "rgba(239,68,68,0.12)"
                  : rateLimitStatus.remaining <= 1
                    ? "rgba(245,158,11,0.12)"
                    : "rgba(129,140,248,0.10)",
                border: `1px solid ${rateLimitBlocked
                  ? "rgba(239,68,68,0.3)"
                  : rateLimitStatus.remaining <= 1
                    ? "rgba(245,158,11,0.3)"
                    : "rgba(129,140,248,0.2)"}`,
                color: rateLimitBlocked
                  ? "#ef4444"
                  : rateLimitStatus.remaining <= 1
                    ? "#f59e0b"
                    : "var(--muted)",
              }}
            >
              <Clock className="w-3 h-3" />
              {rateLimitBlocked
                ? `Resets in ${formatRetry(retryAfter)}`
                : `${rateLimitStatus.remaining} of ${rateLimitStatus.limit} evaluations remaining`}
            </div>
          </motion.div>
        )}

        {/* ── Rate Limit Blocked Screen ────────────────────── */}
        <AnimatePresence>
          {rateLimitBlocked && (
            <motion.div
              key="rate-limit-blocked"
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.97 }}
              className="glass-card rounded-2xl p-10 text-center mb-6"
              style={{ borderColor: "rgba(239,68,68,0.3)" }}
            >
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
                style={{ background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.25)" }}
              >
                <AlertTriangle className="w-8 h-8" style={{ color: "#ef4444" }} />
              </div>
              <h3 className="font-bold text-lg text-[var(--foreground)] mb-2">
                Evaluation Limit Reached
              </h3>
              <p className="text-sm text-[var(--muted)] max-w-sm mx-auto mb-4">
                You&apos;ve used all {rateLimitStatus?.limit} free evaluations for this{" "}
                {rateLimitStatus?.window_hours}-hour window. This keeps the service
                available for everyone.
              </p>
              <div
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold"
                style={{ background: "rgba(239,68,68,0.12)", color: "#ef4444", border: "1px solid rgba(239,68,68,0.25)" }}
              >
                <Clock className="w-4 h-4" />
                Resets in {formatRetry(retryAfter)}
              </div>
              <p className="text-xs text-[var(--muted)] mt-4">
                Already evaluated the same pitch? It&apos;s cached — re-submit to get instant results at no cost.
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Main 2-col grid ─────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Left: Form + Stream */}
          <div className="space-y-5">
            {/* Pitch form card */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              className="glass-card rounded-2xl p-6"
            >
              {/* ── Input mode tabs ── */}
              <div className="flex items-center gap-1 mb-5 p-1 rounded-xl" style={{ background: "var(--hover-bg)" }}>
                {([
                  { key: "text", label: "Type Pitch", icon: Type },
                  { key: "deck", label: "Upload Deck", icon: FileUp },
                ] as const).map(({ key, label, icon: Icon }) => (
                  <button
                    key={key}
                    onClick={() => setInputMode(key)}
                    className="flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all"
                    style={{
                      background: inputMode === key ? "var(--card-bg)" : "transparent",
                      color: inputMode === key ? "var(--primary)" : "var(--muted)",
                      boxShadow: inputMode === key ? "0 1px 3px rgba(0,0,0,0.3)" : "none",
                    }}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    {label}
                  </button>
                ))}
              </div>

              <AnimatePresence mode="wait">
                {inputMode === "text" ? (
                  <motion.div
                    key="text-form"
                    initial={{ opacity: 0, x: -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 12 }}
                    transition={{ duration: 0.2 }}
                  >
                    <PitchForm onSubmit={handleSubmit} isLoading={isLoading} />
                  </motion.div>
                ) : (
                  <motion.div
                    key="deck-upload"
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -12 }}
                    transition={{ duration: 0.2 }}
                  >
                    <DeckUpload onReady={handleSubmit} isLoading={isLoading} />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Agent stream card */}
            <AnimatePresence>
              {(steps.length > 0 || isLoading) && (
                <motion.div
                  key="stream"
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.4 }}
                  className="glass-card rounded-2xl p-6 max-h-[640px] overflow-y-auto"
                >
                  <AgentStream steps={steps} isRunning={isLoading} />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="glass-card rounded-2xl p-4 border border-[var(--danger)]/30"
                >
                  <p className="text-sm text-[var(--danger)]">{error}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right: Results */}
          <div ref={resultsRef}>
            <AnimatePresence mode="wait">
              {evaluation ? (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                  className="glass-card rounded-2xl p-6"
                >
                  <h3 className="font-semibold text-base text-[var(--foreground)] mb-4 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
                      <BarChart3 className="w-3.5 h-3.5 text-white" />
                    </span>
                    Evaluation: {evaluation.startup_name}
                  </h3>
                  <EvaluationResults
                    evaluation={evaluation}
                    processingTime={processingTime ?? undefined}
                    llmProvider={llmProvider ?? undefined}
                    cacheHits={cacheHits}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.15 }}
                  className="glass-card rounded-2xl p-10 text-center h-full flex flex-col items-center justify-center min-h-[300px]"
                >
                  <div
                    className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
                    style={{ background: "linear-gradient(135deg, var(--primary), var(--accent))", opacity: 0.2 }}
                  />
                  <div
                    className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto -mt-20 mb-4"
                    style={{ background: "rgba(129,140,248,0.12)", border: "1px solid rgba(129,140,248,0.2)" }}
                  >
                    <BarChart3 className="w-8 h-8 text-[var(--primary)]" />
                  </div>
                  <h3 className="font-semibold text-[var(--foreground)] mb-2">Results will appear here</h3>
                  <p className="text-sm text-[var(--muted)] max-w-xs">
                    Submit a startup pitch to get a comprehensive AI evaluation with scores, charts, and investment recommendations.
                  </p>

                  {/* How it works */}
                  <div className="mt-8 w-full max-w-xs text-left">
                    <p className="text-xs font-semibold uppercase tracking-widest text-[var(--muted)] mb-3">How it works</p>
                    <div className="space-y-3">
                      {[
                        "Submit your startup pitch",
                        "AI agent researches live market data",
                        `Agent scores ${dimCount} evaluation dimensions`,
                        "Results + charts generated instantly",
                      ].map((text, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <span
                            className="w-6 h-6 rounded-full text-xs font-bold flex items-center justify-center shrink-0 text-[var(--primary)]"
                            style={{ background: "rgba(129,140,248,0.12)" }}
                          >
                            {i + 1}
                          </span>
                          <span className="text-sm text-[var(--muted)]">{text}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 pb-8">
          <p className="text-sm text-[var(--muted)]">
            Built by{" "}
            <a href="https://github.com/Jeet-51" target="_blank" rel="noopener noreferrer"
              className="text-[var(--primary)] hover:underline font-medium">
              Jeet Patel
            </a>{" "}
            · Powered by ReAct Agent · Gemini · Tavily
          </p>
        </footer>
      </main>
    </div>
  );
}
