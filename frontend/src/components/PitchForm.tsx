"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Send, Loader2, Sparkles } from "lucide-react";
import { fetchSamplePitches } from "@/lib/api";
import { SamplePitch } from "@/types";

interface Props {
  onSubmit: (name: string, pitch: string) => void;
  isLoading: boolean;
}

export default function PitchForm({ onSubmit, isLoading }: Props) {
  const [name, setName] = useState("");
  const [pitch, setPitch] = useState("");
  const [samples, setSamples] = useState<SamplePitch[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    fetchSamplePitches()
      .then(setSamples)
      .catch(() =>
        setSamples([
          { name: "HealthAI", pitch: "An AI-powered diagnostic tool that analyzes patient symptoms..." },
          { name: "EduVerse", pitch: "A VR-based learning platform that creates immersive educational..." },
          { name: "GreenCoin", pitch: "A blockchain-based carbon credit marketplace..." },
        ])
      );
  }, []);

  const handleSample = (s: SamplePitch) => {
    setName(s.name);
    setPitch(s.pitch);
    setOpen(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !pitch.trim() || isLoading) return;
    onSubmit(name.trim(), pitch.trim());
  };

  const canSubmit = name.trim() && pitch.trim() && !isLoading;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">

      {/* Sample picker */}
      <div className="relative">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="w-full flex items-center justify-between px-4 py-2.5 glass-input rounded-xl text-sm text-[var(--muted)] hover:text-[var(--foreground)] hover:border-[var(--primary)] transition-all"
        >
          <span className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-[var(--primary)]" />
            Try a sample pitch
          </span>
          <motion.div animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
            <ChevronDown className="w-4 h-4" />
          </motion.div>
        </button>

        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0, y: -6, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6, scale: 0.98 }}
              transition={{ duration: 0.18 }}
              className="absolute top-full left-0 right-0 mt-1.5 rounded-xl overflow-hidden overflow-y-auto z-20 border border-[var(--card-border)]"
              style={{
                background: "var(--bg-secondary)",
                boxShadow: "0 12px 40px rgba(0,0,0,0.2)",
                maxHeight: "240px",
              }}
            >
              {samples.map((s) => (
                <button
                  key={s.name}
                  type="button"
                  onClick={() => handleSample(s)}
                  className="w-full px-4 py-2.5 text-left text-sm hover:bg-[var(--hover-bg)] transition-colors border-b border-[var(--card-border)] last:border-0 overflow-hidden"
                  style={{ maxHeight: "60px" }}
                >
                  <span className="font-medium text-[var(--foreground)]">{s.name}</span>
                  <p className="text-xs text-[var(--muted)] mt-0.5 truncate">{s.pitch.slice(0, 80)}...</p>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Startup name */}
      <div>
        <label className="block text-xs font-semibold text-[var(--muted)] uppercase tracking-wider mb-1.5">
          Startup Name
        </label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. HealthAI"
          className="w-full px-4 py-3 rounded-xl glass-input text-sm"
          disabled={isLoading}
        />
      </div>

      {/* Pitch text */}
      <div>
        <label className="block text-xs font-semibold text-[var(--muted)] uppercase tracking-wider mb-1.5">
          Pitch Description
        </label>
        <textarea
          value={pitch}
          onChange={(e) => setPitch(e.target.value)}
          placeholder="Describe your startup, the problem you're solving, your solution, target market, and traction..."
          rows={6}
          className="w-full px-4 py-3 rounded-xl glass-input text-sm resize-none leading-relaxed"
          disabled={isLoading}
        />
        <p className="text-xs text-[var(--muted)] mt-1 text-right">{pitch.length} chars</p>
      </div>

      {/* Submit */}
      <motion.button
        type="submit"
        disabled={!canSubmit}
        whileHover={canSubmit ? { scale: 1.02 } : {}}
        whileTap={canSubmit ? { scale: 0.98 } : {}}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
        style={{
          background: canSubmit
            ? "linear-gradient(135deg, var(--primary), var(--accent))"
            : "var(--card-border)",
          boxShadow: canSubmit ? "0 0 24px var(--primary-glow)" : "none",
        }}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Agent is thinking…
          </>
        ) : (
          <>
            <Send className="w-4 h-4" />
            Evaluate Pitch
          </>
        )}
      </motion.button>
    </form>
  );
}
