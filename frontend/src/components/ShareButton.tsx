"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Share2, Check, Copy, ExternalLink, Loader2 } from "lucide-react";
import { PitchEvaluation } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Props {
    evaluation: PitchEvaluation;
    startupName: string;
    role: string;
    processingTime?: number;
    llmProvider?: string;
}

export default function ShareButton({
    evaluation,
    startupName,
    role,
    processingTime = 0,
    llmProvider = "unknown",
}: Props) {
    const [state, setState] = useState<"idle" | "loading" | "copied" | "error">("idle");
    const [shareUrl, setShareUrl] = useState<string | null>(null);

    const handleShare = async () => {
        if (shareUrl) {
            // Already have a URL — just copy it
            await navigator.clipboard.writeText(shareUrl);
            setState("copied");
            setTimeout(() => setState("idle"), 2500);
            return;
        }

        setState("loading");
        try {
            const res = await fetch(`${API_BASE}/share`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    evaluation: evaluation,
                    startup_name: startupName,
                    role,
                    processing_time: processingTime,
                    llm_provider: llmProvider,
                }),
            });

            if (!res.ok) throw new Error("Share failed");

            const data = await res.json();
            const url = `${window.location.origin}/eval/${data.share_id}`;
            setShareUrl(url);

            await navigator.clipboard.writeText(url);
            setState("copied");
            setTimeout(() => setState("idle"), 2500);
        } catch {
            setState("error");
            setTimeout(() => setState("idle"), 3000);
        }
    };

    const label = {
        idle: shareUrl ? "Copy Link" : "Share",
        loading: "Creating link…",
        copied: "Link copied!",
        error: "Failed — retry",
    }[state];

    const Icon = {
        idle: shareUrl ? Copy : Share2,
        loading: Loader2,
        copied: Check,
        error: Share2,
    }[state];

    const color = state === "copied"
        ? { bg: "rgba(52,211,153,0.12)", border: "rgba(52,211,153,0.3)", text: "#34d399" }
        : state === "error"
            ? { bg: "rgba(248,113,113,0.12)", border: "rgba(248,113,113,0.3)", text: "#f87171" }
            : { bg: "rgba(129,140,248,0.10)", border: "rgba(129,140,248,0.25)", text: "var(--primary)" };

    return (
        <div className="flex items-center gap-2">
            <motion.button
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                onClick={handleShare}
                disabled={state === "loading"}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all"
                style={{
                    background: color.bg,
                    border: `1px solid ${color.border}`,
                    color: color.text,
                }}
            >
                <Icon
                    className={`w-4 h-4 ${state === "loading" ? "animate-spin" : ""}`}
                />
                {label}
            </motion.button>

            {/* Open link in new tab */}
            <AnimatePresence>
                {shareUrl && (
                    <motion.a
                        key="open-link"
                        initial={{ opacity: 0, scale: 0.85 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.85 }}
                        href={shareUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 rounded-xl text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
                        style={{ background: "var(--hover-bg)", border: "1px solid var(--card-border)" }}
                        title="Open shared link"
                    >
                        <ExternalLink className="w-4 h-4" />
                    </motion.a>
                )}
            </AnimatePresence>
        </div>
    );
}
