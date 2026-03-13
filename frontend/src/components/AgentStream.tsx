"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Search, Eye, CheckCircle2, Terminal } from "lucide-react";
import { AgentStep } from "@/types";

interface Props {
  steps: AgentStep[];
  isRunning: boolean;
}

/* ── Step type config ─────────────────────────────────────── */
const CONFIG = {
  thought: {
    icon: Brain,
    label: "Thought",
    gradient: "from-violet-500 to-purple-600",
    glow: "rgba(139,92,246,0.3)",
    textColor: "text-violet-400",
    bgColor: "rgba(139,92,246,0.07)",
    borderColor: "rgba(139,92,246,0.2)",
  },
  action: {
    icon: Search,
    label: "Action",
    gradient: "from-blue-500 to-cyan-500",
    glow: "rgba(59,130,246,0.3)",
    textColor: "text-blue-400",
    bgColor: "rgba(59,130,246,0.07)",
    borderColor: "rgba(59,130,246,0.2)",
  },
  observation: {
    icon: Eye,
    label: "Observation",
    gradient: "from-amber-500 to-orange-500",
    glow: "rgba(245,158,11,0.3)",
    textColor: "text-amber-400",
    bgColor: "rgba(245,158,11,0.07)",
    borderColor: "rgba(245,158,11,0.2)",
  },
  final_answer: {
    icon: CheckCircle2,
    label: "Final Answer",
    gradient: "from-emerald-500 to-teal-500",
    glow: "rgba(16,185,129,0.3)",
    textColor: "text-emerald-400",
    bgColor: "rgba(16,185,129,0.07)",
    borderColor: "rgba(16,185,129,0.2)",
  },
} as const;

/* ── Step card ────────────────────────────────────────────── */
function StepCard({ step, index }: { step: AgentStep; index: number }) {
  const cfg = CONFIG[step.step_type as keyof typeof CONFIG] ?? CONFIG.thought;
  const Icon = cfg.icon;

  return (
    <motion.div
      initial={{ opacity: 0, x: -16, scale: 0.98 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1], delay: index * 0.04 }}
      className="relative flex gap-3"
    >
      {/* Timeline dot + line */}
      <div className="flex flex-col items-center shrink-0">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.25, delay: index * 0.04 + 0.1, type: "spring", stiffness: 300 }}
          className={`w-8 h-8 rounded-xl bg-gradient-to-br ${cfg.gradient} flex items-center justify-center shrink-0 z-10`}
          style={{ boxShadow: `0 0 12px ${cfg.glow}` }}
        >
          <Icon className="w-4 h-4 text-white" />
        </motion.div>
        {/* Connector line (not on last item) */}
        <div className="w-px flex-1 mt-1" style={{ background: cfg.borderColor, minHeight: "8px" }} />
      </div>

      {/* Content card */}
      <div
        className="flex-1 rounded-xl p-3.5 mb-3 min-w-0"
        style={{ background: cfg.bgColor, border: `1px solid ${cfg.borderColor}` }}
      >
        <div className="flex items-center gap-2 mb-1.5">
          <span className={`text-xs font-bold uppercase tracking-widest ${cfg.textColor}`}>
            {cfg.label}
          </span>
          {step.tool_name && (
            <span className="text-xs px-2 py-0.5 rounded-full font-mono"
              style={{ background: cfg.borderColor, color: cfg.textColor.replace("400", "300") }}>
              {step.tool_name}
            </span>
          )}
          <span className="text-xs text-[var(--muted)] ml-auto">#{index + 1}</span>
        </div>
        <p className="text-sm text-[var(--foreground)] leading-relaxed whitespace-pre-wrap break-words">
          {step.content}
        </p>
        {step.tool_input && (
          <div className="mt-2 px-3 py-2 rounded-lg font-mono text-xs text-[var(--muted)] break-all"
            style={{ background: "rgba(0,0,0,0.15)" }}>
            {step.tool_input}
          </div>
        )}
      </div>
    </motion.div>
  );
}

/* ── Thinking dots ────────────────────────────────────────── */
function ThinkingDots() {
  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 12 }}
      className="flex items-center gap-3 mt-1"
    >
      <div className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0"
        style={{ background: "rgba(129,140,248,0.12)", border: "1px solid rgba(129,140,248,0.2)" }}>
        <Brain className="w-4 h-4 text-[var(--primary)] animate-pulse" />
      </div>
      <div className="flex-1 rounded-xl p-3.5"
        style={{ background: "rgba(129,140,248,0.06)", border: "1px solid rgba(129,140,248,0.15)" }}>
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-[var(--muted)] mr-1">Agent reasoning</span>
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-[var(--primary)]"
              animate={{ opacity: [0.2, 1, 0.2] }}
              transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.3 }}
            />
          ))}
        </div>
      </div>
    </motion.div>
  );
}

/* ── Main component ───────────────────────────────────────── */
export default function AgentStream({ steps, isRunning }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  /* Auto-scroll to latest step */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps.length]);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-2.5 mb-5">
        <div className="w-7 h-7 rounded-lg flex items-center justify-center"
          style={{ background: "linear-gradient(135deg, var(--primary), var(--accent))" }}>
          <Terminal className="w-3.5 h-3.5 text-white" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-[var(--foreground)]">Agent Reasoning</h3>
          <p className="text-xs text-[var(--muted)]">{steps.length} step{steps.length !== 1 ? "s" : ""} completed</p>
        </div>

        {/* Live indicator */}
        {isRunning && (
          <div className="ml-auto flex items-center gap-1.5 px-2.5 py-1 rounded-full"
            style={{ background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.25)" }}>
            <motion.div
              className="w-1.5 h-1.5 rounded-full bg-[var(--success)]"
              animate={{ scale: [1, 1.4, 1], opacity: [1, 0.6, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            />
            <span className="text-xs font-medium text-[var(--success)]">Live</span>
          </div>
        )}
      </div>

      {/* Steps */}
      <div>
        <AnimatePresence initial={false}>
          {steps.map((step, i) => (
            <StepCard key={`${step.step_number}-${step.step_type}`} step={step} index={i} />
          ))}
        </AnimatePresence>

        {/* Thinking indicator */}
        <AnimatePresence>
          {isRunning && <ThinkingDots key="thinking" />}
        </AnimatePresence>

        <div ref={bottomRef} />
      </div>

      {/* Completion badge */}
      <AnimatePresence>
        {!isRunning && steps.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mt-4 flex items-center gap-2 px-4 py-2.5 rounded-xl"
            style={{ background: "rgba(52,211,153,0.08)", border: "1px solid rgba(52,211,153,0.22)" }}
          >
            <CheckCircle2 className="w-4 h-4 text-[var(--success)] shrink-0" />
            <span className="text-sm font-medium text-[var(--success)]">
              Analysis complete · {steps.length} reasoning steps
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
