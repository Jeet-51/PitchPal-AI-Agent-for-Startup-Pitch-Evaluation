"use client";

import {
  Radar, RadarChart as RechartsRadar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip,
} from "recharts";
import { PitchEvaluation } from "@/types";

export default function RadarChart({ evaluation }: { evaluation: PitchEvaluation }) {
  const data = evaluation.dimensions.map((d) => ({
    dimension: d.name.length > 16 ? d.name.slice(0, 14) + "…" : d.name,
    score: d.score,
    fullMark: 10,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <RechartsRadar data={data} cx="50%" cy="50%" outerRadius="72%">
        <PolarGrid stroke="var(--card-border)" />
        <PolarAngleAxis
          dataKey="dimension"
          tick={{ fill: "var(--muted)", fontSize: 10, fontFamily: "inherit" }}
        />
        <PolarRadiusAxis
          angle={90} domain={[0, 10]} tickCount={6}
          tick={{ fill: "var(--muted)", fontSize: 9 }}
          stroke="transparent"
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
          formatter={(v) => [Number(v).toFixed(1), "Score"]}
        />
        <Radar
          name={evaluation.startup_name}
          dataKey="score"
          stroke="var(--primary)"
          fill="var(--primary)"
          fillOpacity={0.18}
          strokeWidth={2}
        />
      </RechartsRadar>
    </ResponsiveContainer>
  );
}
