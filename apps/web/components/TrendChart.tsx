"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RunSummary } from "@/lib/api";

export function TrendChart({ runs }: { runs: RunSummary[] }) {
  const data = [...runs].reverse().map((r, i) => ({
    name: `#${i + 1}`,
    recall: Number(r.retrieval_metrics.recall.toFixed(3)),
    ndcg: Number(r.retrieval_metrics.ndcg.toFixed(3)),
  }));

  return (
    <div className="chart-panel">
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -18 }}>
          <CartesianGrid stroke="#1f2a3d" strokeDasharray="3 3" />
          <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
          <YAxis domain={[0, 1]} stroke="#64748b" fontSize={12} />
          <Tooltip
            contentStyle={{
              background: "#111827",
              border: "1px solid #1f2a3d",
              borderRadius: 8,
              fontSize: 13,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 13 }} />
          <Line type="monotone" dataKey="recall" stroke="#2dd4bf" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="ndcg" stroke="#a78bfa" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
