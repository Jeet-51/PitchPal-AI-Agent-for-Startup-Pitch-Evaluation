"use client";

import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import { PitchEvaluation } from "@/types";

function barColor(score: number) {
  if (score >= 7) return "#34d399";
  if (score >= 5) return "#fbbf24";
  return "#f87171";
}

export default function ScoreBarChart({ evaluation }: { evaluation: PitchEvaluation }) {
  const data = evaluation.dimensions.map((d) => ({
    name: d.name.length > 18 ? d.name.slice(0, 16) + "…" : d.name,
    score: d.score,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} layout="vertical" margin={{ left: 16, right: 20, top: 4, bottom: 4 }}>
        <XAxis
          type="number" domain={[0, 10]}
          tick={{ fontSize: 11, fill: "var(--muted)", fontFamily: "inherit" }}
          axisLine={false} tickLine={false}
        />
        <YAxis
          type="category" dataKey="name" width={130}
          tick={{ fontSize: 10, fill: "var(--muted)", fontFamily: "inherit" }}
          axisLine={false} tickLine={false}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "var(--bg-secondary)",
            border: "1px solid var(--card-border)",
            borderRadius: "10px",
            fontSize: "12px",
            color: "var(--foreground)",
            boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
          }}
          cursor={{ fill: "var(--hover-bg)" }}
          formatter={(v) => [Number(v).toFixed(1), "Score"]}
        />
        <Bar dataKey="score" radius={[0, 6, 6, 0]} barSize={22}>
          {data.map((entry, i) => (
            <Cell key={i} fill={barColor(entry.score)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
