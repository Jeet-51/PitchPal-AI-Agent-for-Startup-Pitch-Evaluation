"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Upload, FileText, CheckCircle2, AlertCircle,
    Loader2, X, ArrowRight, Palette, BookOpen, BarChart2,
} from "lucide-react";
import { DeckUploadResponse } from "@/types";
import { uploadDeck } from "@/lib/api";

interface Props {
    onReady: (startupName: string, pitchText: string) => void;
    isLoading: boolean;
}

function ScorePill({
    label, score, icon: Icon, feedback,
}: {
    label: string;
    score: number;
    icon: React.ElementType;
    feedback: string;
}) {
    const color =
        score >= 7 ? "#34d399"
            : score >= 5 ? "#fbbf24"
                : score > 0 ? "#f87171"
                    : "var(--muted)";

    return (
        <div
            className="rounded-xl p-3 space-y-2"
            style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--card-border)" }}
        >
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Icon className="w-3.5 h-3.5" style={{ color }} />
                    <span className="text-xs font-semibold text-[var(--foreground)]">{label}</span>
                </div>
                <span className="text-sm font-bold" style={{ color }}>
                    {score > 0 ? score.toFixed(1) : "—"}
                </span>
            </div>
            {/* Mini progress bar */}
            <div className="h-1 rounded-full overflow-hidden" style={{ background: "var(--card-border)" }}>
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(score / 10) * 100}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="h-full rounded-full"
                    style={{ background: color }}
                />
            </div>
            <p className="text-[11px] text-[var(--muted)] leading-relaxed">{feedback}</p>
        </div>
    );
}

export default function DeckUpload({ onReady, isLoading }: Props) {
    const [isDragging, setIsDragging] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<DeckUploadResponse | null>(null);
    const [startupName, setStartupName] = useState("");
    const fileInputRef = useRef<HTMLInputElement>(null);

    const ACCEPTED = ".pdf,.pptx,.ppt";
    const MAX_MB = 20;

    const processFile = useCallback(async (file: File) => {
        setError(null);
        setResult(null);

        // Validate
        const ext = file.name.toLowerCase().split(".").pop() ?? "";
        if (!["pdf", "pptx", "ppt"].includes(ext)) {
            setError("Unsupported format. Please upload a PDF or PPTX file.");
            return;
        }
        if (file.size > MAX_MB * 1024 * 1024) {
            setError(`File too large. Maximum size is ${MAX_MB} MB.`);
            return;
        }

        setIsAnalyzing(true);
        try {
            const data = await uploadDeck(file);
            setResult(data);
            setStartupName(data.startup_name !== "Unknown" ? data.startup_name : "");
        } catch (err) {
            setError(err instanceof Error ? err.message : "Upload failed. Please try again.");
        } finally {
            setIsAnalyzing(false);
        }
    }, []);

    const onDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setIsDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) processFile(file);
        },
        [processFile]
    );

    const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) processFile(file);
    };

    const reset = () => {
        setResult(null);
        setError(null);
        setStartupName("");
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const handleEvaluate = () => {
        if (!result) return;
        const name = startupName.trim() || result.startup_name;
        onReady(name, result.extracted_text);
    };

    return (
        <div className="space-y-4">
            {/* Drop zone — shown when no result yet */}
            <AnimatePresence mode="wait">
                {!result ? (
                    <motion.div
                        key="dropzone"
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        transition={{ duration: 0.3 }}
                    >
                        <div
                            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                            onDragLeave={() => setIsDragging(false)}
                            onDrop={onDrop}
                            onClick={() => !isAnalyzing && fileInputRef.current?.click()}
                            className="relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all"
                            style={{
                                borderColor: isDragging ? "var(--primary)" : "var(--card-border)",
                                background: isDragging
                                    ? "rgba(129,140,248,0.08)"
                                    : "rgba(255,255,255,0.02)",
                            }}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept={ACCEPTED}
                                onChange={onFileChange}
                                className="hidden"
                            />

                            <AnimatePresence mode="wait">
                                {isAnalyzing ? (
                                    <motion.div
                                        key="loading"
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.9 }}
                                        className="space-y-3"
                                    >
                                        <div className="w-12 h-12 rounded-full bg-[rgba(129,140,248,0.15)] flex items-center justify-center mx-auto">
                                            <Loader2 className="w-6 h-6 text-[var(--primary)] animate-spin" />
                                        </div>
                                        <div>
                                            <p className="text-sm font-semibold text-[var(--foreground)]">
                                                Analyzing your deck...
                                            </p>
                                            <p className="text-xs text-[var(--muted)] mt-1">
                                                Gemini Vision is reviewing your slides
                                            </p>
                                        </div>
                                    </motion.div>
                                ) : (
                                    <motion.div
                                        key="idle"
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.9 }}
                                        className="space-y-3"
                                    >
                                        <div
                                            className="w-12 h-12 rounded-full flex items-center justify-center mx-auto"
                                            style={{ background: "rgba(129,140,248,0.12)" }}
                                        >
                                            <Upload className="w-6 h-6 text-[var(--primary)]" />
                                        </div>
                                        <div>
                                            <p className="text-sm font-semibold text-[var(--foreground)]">
                                                Drop your pitch deck here
                                            </p>
                                            <p className="text-xs text-[var(--muted)] mt-1">
                                                PDF or PPTX · Max {MAX_MB} MB · up to 20 slides
                                            </p>
                                        </div>
                                        <span
                                            className="inline-block text-xs px-3 py-1.5 rounded-lg font-medium"
                                            style={{ background: "rgba(129,140,248,0.15)", color: "var(--primary)" }}
                                        >
                                            Click to browse
                                        </span>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>

                        {/* Error */}
                        <AnimatePresence>
                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, y: 4 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0 }}
                                    className="flex items-center gap-2 text-sm mt-2 px-3 py-2 rounded-lg"
                                    style={{ background: "rgba(248,113,113,0.08)", color: "var(--danger)" }}
                                >
                                    <AlertCircle className="w-4 h-4 shrink-0" />
                                    {error}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                ) : (
                    /* ─── Analysis results ────────────────────────────── */
                    <motion.div
                        key="results"
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="space-y-4"
                    >
                        {/* File info header */}
                        <div
                            className="flex items-center justify-between px-4 py-3 rounded-xl"
                            style={{ background: "rgba(52,211,153,0.06)", border: "1px solid rgba(52,211,153,0.2)" }}
                        >
                            <div className="flex items-center gap-3">
                                <CheckCircle2 className="w-4 h-4 text-[var(--success)]" />
                                <div>
                                    <p className="text-sm font-semibold text-[var(--foreground)]">
                                        {result.slide_count} slides analyzed
                                    </p>
                                    <p className="text-xs text-[var(--muted)]">
                                        {result.file_format.toUpperCase()} · Gemini Vision{" "}
                                        {result.file_format === "pptx" ? "(text-only)" : "complete"}
                                    </p>
                                </div>
                            </div>
                            <button
                                onClick={reset}
                                className="p-1.5 rounded-lg hover:bg-[var(--hover-bg)] transition-colors"
                                title="Upload a different deck"
                            >
                                <X className="w-3.5 h-3.5 text-[var(--muted)]" />
                            </button>
                        </div>

                        {/* Overall deck score */}
                        <div className="text-center">
                            <p className="text-xs font-semibold uppercase tracking-widest text-[var(--muted)] mb-1">
                                Deck Quality Score
                            </p>
                            <p className="text-4xl font-black" style={{
                                color: result.deck_quality.overall_deck_score >= 7 ? "#34d399"
                                    : result.deck_quality.overall_deck_score >= 5 ? "#fbbf24" : "#f87171"
                            }}>
                                {result.deck_quality.overall_deck_score > 0
                                    ? result.deck_quality.overall_deck_score.toFixed(1)
                                    : "—"}
                                <span className="text-xl font-medium text-[var(--muted)]"> / 10</span>
                            </p>
                        </div>

                        {/* Score pills */}
                        <div className="space-y-2">
                            <ScorePill
                                label="Design Quality"
                                score={result.deck_quality.design_score}
                                icon={Palette}
                                feedback={result.deck_quality.design_feedback}
                            />
                            <ScorePill
                                label="Narrative Flow"
                                score={result.deck_quality.narrative_score}
                                icon={BookOpen}
                                feedback={result.deck_quality.narrative_feedback}
                            />
                            <ScorePill
                                label="Data & Charts"
                                score={result.deck_quality.data_viz_score}
                                icon={BarChart2}
                                feedback={result.deck_quality.data_viz_feedback}
                            />
                        </div>

                        {/* Strengths + Improvements */}
                        {(result.deck_quality.strengths.length > 0 || result.deck_quality.improvements.length > 0) && (
                            <div className="grid grid-cols-2 gap-3">
                                {result.deck_quality.strengths.length > 0 && (
                                    <div className="space-y-1.5 p-3 rounded-xl" style={{ background: "rgba(52,211,153,0.06)", border: "1px solid rgba(52,211,153,0.15)" }}>
                                        <p className="text-xs font-bold text-[var(--success)]">Strengths</p>
                                        {result.deck_quality.strengths.map((s, i) => (
                                            <div key={i} className="flex items-start gap-1.5 text-xs text-[var(--foreground)]">
                                                <span className="w-1 h-1 rounded-full bg-[var(--success)] mt-1.5 shrink-0" />
                                                {s}
                                            </div>
                                        ))}
                                    </div>
                                )}
                                {result.deck_quality.improvements.length > 0 && (
                                    <div className="space-y-1.5 p-3 rounded-xl" style={{ background: "rgba(251,191,36,0.06)", border: "1px solid rgba(251,191,36,0.15)" }}>
                                        <p className="text-xs font-bold text-[var(--warning)]">Improve</p>
                                        {result.deck_quality.improvements.map((s, i) => (
                                            <div key={i} className="flex items-start gap-1.5 text-xs text-[var(--foreground)]">
                                                <span className="w-1 h-1 rounded-full bg-[var(--warning)] mt-1.5 shrink-0" />
                                                {s}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Startup name input */}
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-[var(--muted)]">
                                <FileText className="inline w-3 h-3 mr-1" />
                                Startup Name
                            </label>
                            <input
                                type="text"
                                value={startupName}
                                onChange={(e) => setStartupName(e.target.value)}
                                placeholder={result.startup_name || "Enter startup name"}
                                className="w-full px-3 py-2 rounded-xl text-sm bg-[var(--input-bg)] border border-[var(--card-border)] text-[var(--foreground)] placeholder:text-[var(--muted)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                            />
                        </div>

                        {/* Evaluate button */}
                        <motion.button
                            whileHover={{ scale: 1.01 }}
                            whileTap={{ scale: 0.99 }}
                            onClick={handleEvaluate}
                            disabled={isLoading}
                            className="w-full py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2 transition-all disabled:opacity-60"
                            style={{
                                background: "linear-gradient(135deg, var(--primary), var(--accent))",
                                color: "white",
                            }}
                        >
                            {isLoading ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                                <>
                                    <span>Evaluate with AI Research</span>
                                    <ArrowRight className="w-4 h-4" />
                                </>
                            )}
                        </motion.button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
