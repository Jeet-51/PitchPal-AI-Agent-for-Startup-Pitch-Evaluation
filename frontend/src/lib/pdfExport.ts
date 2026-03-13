import { jsPDF } from "jspdf";
import { PitchEvaluation, DimensionScore } from "@/types";

// ── Color Palette ──────────────────────────────────────────
const COLORS = {
  primary: [37, 99, 235] as [number, number, number],       // blue-600
  primaryLight: [59, 130, 246] as [number, number, number],  // blue-500
  dark: [15, 23, 42] as [number, number, number],            // slate-900
  text: [30, 41, 59] as [number, number, number],            // slate-800
  muted: [100, 116, 139] as [number, number, number],        // slate-500
  light: [241, 245, 249] as [number, number, number],        // slate-100
  white: [255, 255, 255] as [number, number, number],
  green: [22, 163, 74] as [number, number, number],
  greenBg: [240, 253, 244] as [number, number, number],
  amber: [217, 119, 6] as [number, number, number],
  amberBg: [255, 251, 235] as [number, number, number],
  red: [220, 38, 38] as [number, number, number],
  redBg: [254, 242, 242] as [number, number, number],
  blueBg: [239, 246, 255] as [number, number, number],
};

function getScoreColor(score: number): [number, number, number] {
  if (score >= 7) return COLORS.green;
  if (score >= 5) return COLORS.amber;
  return COLORS.red;
}

function getScoreBg(score: number): [number, number, number] {
  if (score >= 7) return COLORS.greenBg;
  if (score >= 5) return COLORS.amberBg;
  return COLORS.redBg;
}

function getRecLabel(rec: string): { color: [number, number, number]; bg: [number, number, number] } {
  const r = rec.toLowerCase();
  if (r.includes("strong buy")) return { color: [6, 95, 70], bg: [209, 250, 229] };
  if (r.includes("buy")) return { color: [22, 101, 52], bg: [220, 252, 231] };
  if (r.includes("hold")) return { color: [146, 64, 14], bg: [254, 243, 199] };
  return { color: [153, 27, 27], bg: [254, 226, 226] };
}

// ── Helper: wrapped text with auto page-break ──────────────
function addWrappedText(
  pdf: jsPDF,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number
): number {
  const lines = pdf.splitTextToSize(text, maxWidth);
  for (const line of lines) {
    if (y > 275) {
      pdf.addPage();
      y = 20;
    }
    pdf.text(line, x, y);
    y += lineHeight;
  }
  return y;
}

// ── Helper: draw a rounded rectangle ───────────────────────
function drawRoundedRect(
  pdf: jsPDF,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
  fillColor: [number, number, number],
  borderColor?: [number, number, number]
) {
  pdf.setFillColor(...fillColor);
  if (borderColor) {
    pdf.setDrawColor(...borderColor);
    pdf.setLineWidth(0.3);
  }
  pdf.roundedRect(x, y, w, h, r, r, borderColor ? "FD" : "F");
}

// ── Helper: check page break ──────────────────────────────
function checkPageBreak(pdf: jsPDF, y: number, needed: number): number {
  if (y + needed > 275) {
    pdf.addPage();
    return 20;
  }
  return y;
}

// ── Main Export Function ───────────────────────────────────
export async function exportPitchPDF(evaluation: PitchEvaluation, processingTime?: number, llmProvider?: string) {
  const pdf = new jsPDF("p", "mm", "a4");
  const pageWidth = pdf.internal.pageSize.getWidth(); // 210
  const margin = 15;
  const contentWidth = pageWidth - margin * 2;
  let y = 0;

  // ══════════════════════════════════════════════════════════
  // COVER / HEADER SECTION
  // ══════════════════════════════════════════════════════════

  // Blue header bar
  pdf.setFillColor(...COLORS.primary);
  pdf.rect(0, 0, pageWidth, 45, "F");

  // Logo text
  pdf.setTextColor(...COLORS.white);
  pdf.setFontSize(24);
  pdf.setFont("helvetica", "bold");
  pdf.text("PitchPal", margin, 18);

  pdf.setFontSize(10);
  pdf.setFont("helvetica", "normal");
  pdf.text("AI-Powered Startup Pitch Evaluation Report", margin, 26);

  // Startup name
  pdf.setFontSize(16);
  pdf.setFont("helvetica", "bold");
  pdf.text(evaluation.startup_name, margin, 38);

  // Date on right side
  pdf.setFontSize(9);
  pdf.setFont("helvetica", "normal");
  const dateStr = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  pdf.text(dateStr, pageWidth - margin, 38, { align: "right" });

  y = 55;

  // ══════════════════════════════════════════════════════════
  // EXECUTIVE SUMMARY — Score + Recommendation
  // ══════════════════════════════════════════════════════════

  pdf.setFontSize(13);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(...COLORS.dark);
  pdf.text("Executive Summary", margin, y);
  y += 8;

  // Overall Score box
  const scoreBoxW = 55;
  const scoreBoxH = 32;
  drawRoundedRect(pdf, margin, y, scoreBoxW, scoreBoxH, 3, getScoreBg(evaluation.overall_score));

  pdf.setFontSize(28);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(...getScoreColor(evaluation.overall_score));
  pdf.text(evaluation.overall_score.toFixed(1), margin + scoreBoxW / 2, y + 18, { align: "center" });

  pdf.setFontSize(8);
  pdf.setFont("helvetica", "normal");
  pdf.setTextColor(...COLORS.muted);
  pdf.text("OVERALL SCORE (out of 10)", margin + scoreBoxW / 2, y + 26, { align: "center" });

  // Recommendation box
  const recBoxX = margin + scoreBoxW + 8;
  const recBoxW = contentWidth - scoreBoxW - 8;
  const rec = getRecLabel(evaluation.investment_recommendation);
  drawRoundedRect(pdf, recBoxX, y, recBoxW, scoreBoxH, 3, rec.bg);

  pdf.setFontSize(18);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(...rec.color);
  pdf.text(evaluation.investment_recommendation, recBoxX + recBoxW / 2, y + 18, { align: "center" });

  pdf.setFontSize(8);
  pdf.setFont("helvetica", "normal");
  pdf.setTextColor(...COLORS.muted);
  pdf.text("INVESTMENT RECOMMENDATION", recBoxX + recBoxW / 2, y + 26, { align: "center" });

  y += scoreBoxH + 12;

  // ══════════════════════════════════════════════════════════
  // DIMENSION SCORES TABLE
  // ══════════════════════════════════════════════════════════

  y = checkPageBreak(pdf, y, 60);

  pdf.setFontSize(13);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(...COLORS.dark);
  pdf.text("Dimension Scores", margin, y);
  y += 8;

  const dimensions: DimensionScore[] = evaluation.dimensions;

  // Table header
  drawRoundedRect(pdf, margin, y, contentWidth, 8, 1, COLORS.dark);
  pdf.setFontSize(8);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(...COLORS.white);
  pdf.text("DIMENSION", margin + 4, y + 5.5);
  pdf.text("SCORE", margin + 75, y + 5.5);
  pdf.text("RATING", margin + 95, y + 5.5);
  pdf.text("BAR", margin + 115, y + 5.5);
  y += 10;

  // Table rows
  for (const dim of dimensions) {
    const rowH = 9;
    y = checkPageBreak(pdf, y, rowH + 2);

    // Alternating row background
    drawRoundedRect(pdf, margin, y, contentWidth, rowH, 1, COLORS.light);

    pdf.setFontSize(9);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(...COLORS.text);
    pdf.text(dim.name, margin + 4, y + 6);

    // Score
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(...getScoreColor(dim.score));
    pdf.text(dim.score.toFixed(1), margin + 77, y + 6);

    // Rating label
    let rating = "Poor";
    if (dim.score >= 8) rating = "Excellent";
    else if (dim.score >= 7) rating = "Good";
    else if (dim.score >= 5) rating = "Average";
    else if (dim.score >= 3) rating = "Below Avg";

    pdf.setFontSize(8);
    pdf.setFont("helvetica", "normal");
    pdf.text(rating, margin + 95, y + 6);

    // Score bar
    const barX = margin + 115;
    const barW = contentWidth - 120;
    const barH = 4;
    const barY = y + 3;
    pdf.setFillColor(220, 220, 220);
    pdf.roundedRect(barX, barY, barW, barH, 1, 1, "F");
    pdf.setFillColor(...getScoreColor(dim.score));
    const filledW = Math.max(2, (dim.score / 10) * barW);
    pdf.roundedRect(barX, barY, filledW, barH, 1, 1, "F");

    y += rowH + 1;
  }

  y += 8;

  // ══════════════════════════════════════════════════════════
  // DETAILED ANALYSIS (per dimension)
  // ══════════════════════════════════════════════════════════

  y = checkPageBreak(pdf, y, 30);

  pdf.setFontSize(13);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(...COLORS.dark);
  pdf.text("Detailed Analysis", margin, y);
  y += 3;

  for (const dim of dimensions) {
    // Estimate space needed
    y = checkPageBreak(pdf, y, 35);
    y += 7;

    // Dimension header with score badge
    const scoreColor = getScoreColor(dim.score);
    const scoreBg = getScoreBg(dim.score);

    // Score badge
    drawRoundedRect(pdf, margin, y, 14, 8, 2, scoreBg);
    pdf.setFontSize(10);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(...scoreColor);
    pdf.text(dim.score.toFixed(1), margin + 7, y + 5.8, { align: "center" });

    // Dimension name
    pdf.setFontSize(11);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(...COLORS.dark);
    pdf.text(dim.name, margin + 18, y + 5.8);

    y += 12;

    // Reasoning
    pdf.setFontSize(8);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(...COLORS.muted);
    pdf.text("ANALYSIS", margin + 2, y);
    y += 4;

    pdf.setFontSize(9);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(...COLORS.text);
    y = addWrappedText(pdf, dim.reasoning, margin + 2, y, contentWidth - 4, 4.5);

    y += 2;

    // Suggestions
    if (dim.suggestions.length > 0) {
      y = checkPageBreak(pdf, y, 15);

      pdf.setFontSize(8);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(...COLORS.muted);
      pdf.text("SUGGESTIONS", margin + 2, y);
      y += 4;

      pdf.setFontSize(9);
      pdf.setFont("helvetica", "normal");
      pdf.setTextColor(...COLORS.primary);

      for (const suggestion of dim.suggestions) {
        y = checkPageBreak(pdf, y, 8);
        pdf.text("→", margin + 3, y);
        y = addWrappedText(pdf, suggestion, margin + 9, y, contentWidth - 12, 4.5);
        y += 1;
      }
    }

    // Separator line
    y += 2;
    y = checkPageBreak(pdf, y, 5);
    pdf.setDrawColor(220, 220, 220);
    pdf.setLineWidth(0.2);
    pdf.line(margin, y, pageWidth - margin, y);
    y += 2;
  }

  y += 5;

  // ══════════════════════════════════════════════════════════
  // KEY STRENGTHS
  // ══════════════════════════════════════════════════════════

  y = checkPageBreak(pdf, y, 30);

  // Section header
  drawRoundedRect(pdf, margin, y, contentWidth, 8, 2, COLORS.greenBg, [187, 247, 208]);
  pdf.setFontSize(10);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(...COLORS.green);
  pdf.text("✓  Key Strengths", margin + 4, y + 5.8);
  y += 12;

  pdf.setFontSize(9);
  pdf.setFont("helvetica", "normal");
  pdf.setTextColor(21, 128, 61); // green-700

  for (const strength of evaluation.key_strengths) {
    y = checkPageBreak(pdf, y, 10);
    pdf.setFillColor(...COLORS.green);
    pdf.circle(margin + 4, y - 1, 1, "F");
    y = addWrappedText(pdf, strength, margin + 9, y, contentWidth - 12, 4.5);
    y += 2;
  }

  y += 6;

  // ══════════════════════════════════════════════════════════
  // MAIN CONCERNS
  // ══════════════════════════════════════════════════════════

  y = checkPageBreak(pdf, y, 30);

  drawRoundedRect(pdf, margin, y, contentWidth, 8, 2, COLORS.amberBg, [253, 230, 138]);
  pdf.setFontSize(10);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(...COLORS.amber);
  pdf.text("⚠  Main Concerns", margin + 4, y + 5.8);
  y += 12;

  pdf.setFontSize(9);
  pdf.setFont("helvetica", "normal");
  pdf.setTextColor(180, 83, 9); // amber-700

  for (const concern of evaluation.main_concerns) {
    y = checkPageBreak(pdf, y, 10);
    pdf.setFillColor(...COLORS.amber);
    pdf.circle(margin + 4, y - 1, 1, "F");
    y = addWrappedText(pdf, concern, margin + 9, y, contentWidth - 12, 4.5);
    y += 2;
  }

  y += 6;

  // ══════════════════════════════════════════════════════════
  // NEXT STEPS
  // ══════════════════════════════════════════════════════════

  y = checkPageBreak(pdf, y, 30);

  drawRoundedRect(pdf, margin, y, contentWidth, 8, 2, COLORS.blueBg, [191, 219, 254]);
  pdf.setFontSize(10);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(...COLORS.primary);
  pdf.text("→  Recommended Next Steps", margin + 4, y + 5.8);
  y += 12;

  pdf.setFontSize(9);
  pdf.setFont("helvetica", "normal");
  pdf.setTextColor(29, 78, 216); // blue-700

  for (let i = 0; i < evaluation.next_steps.length; i++) {
    y = checkPageBreak(pdf, y, 10);
    pdf.setFont("helvetica", "bold");
    pdf.text(`${i + 1}.`, margin + 3, y);
    pdf.setFont("helvetica", "normal");
    y = addWrappedText(pdf, evaluation.next_steps[i], margin + 10, y, contentWidth - 13, 4.5);
    y += 2;
  }

  y += 8;

  // ══════════════════════════════════════════════════════════
  // FOOTER
  // ══════════════════════════════════════════════════════════

  y = checkPageBreak(pdf, y, 20);

  pdf.setDrawColor(200, 200, 200);
  pdf.setLineWidth(0.3);
  pdf.line(margin, y, pageWidth - margin, y);
  y += 6;

  pdf.setFontSize(8);
  pdf.setFont("helvetica", "normal");
  pdf.setTextColor(...COLORS.muted);

  const metaParts: string[] = [];
  if (processingTime) metaParts.push(`Processing time: ${processingTime.toFixed(1)}s`);
  if (llmProvider) metaParts.push(`LLM: ${llmProvider}`);
  metaParts.push(`Generated: ${dateStr}`);

  const dateStr2 = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  pdf.text(`Generated by PitchPal AI Agent  •  ${dateStr2}`, margin, y);
  y += 4;

  const techLine: string[] = [];
  if (processingTime) techLine.push(`Processing time: ${processingTime.toFixed(1)}s`);
  if (llmProvider) techLine.push(`Powered by ${llmProvider}`);
  techLine.push("Real-time web research via Tavily Search API");
  pdf.text(techLine.join("  •  "), margin, y);

  y += 4;
  pdf.setFontSize(7);
  pdf.text(
    "This report was generated using a ReAct (Reasoning + Acting) AI agent that performed live web research to evaluate this pitch.",
    margin,
    y
  );

  // ── Page numbers on all pages ────────────────────────────
  const totalPages = pdf.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    pdf.setPage(i);
    pdf.setFontSize(8);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(...COLORS.muted);
    pdf.text(`Page ${i} of ${totalPages}`, pageWidth / 2, 290, { align: "center" });

    // Thin top accent line on pages after the first
    if (i > 1) {
      pdf.setFillColor(...COLORS.primary);
      pdf.rect(0, 0, pageWidth, 2, "F");
    }
  }

  // ── Save ─────────────────────────────────────────────────
  const safeName = evaluation.startup_name.replace(/[^a-zA-Z0-9]/g, "_");
  pdf.save(`PitchPal_${safeName}_Report.pdf`);
}
