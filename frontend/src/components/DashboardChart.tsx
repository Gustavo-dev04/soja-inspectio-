"use client";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

const COLORS = ["#22c55e", "#84cc16", "#f59e0b", "#ef4444", "#8b5cf6"];

interface Props {
  data: { name: string; value: number }[];
  type?: "pie" | "bar";
}

export default function DashboardChart({ data, type = "pie" }: Props) {
  if (type === "bar") {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          margin={{ top: 10, right: 20, left: 0, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="name"
            angle={-35}
            textAnchor="end"
            tick={{ fontSize: 11 }}
          />
          <YAxis unit="%" />
          <Tooltip
            formatter={(value: number | string) =>
              `${Number(value).toFixed(1)}%`
            }
          />
          <Bar dataKey="value" name="Percentual">
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={100}
          label
        >
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          formatter={(value: number | string) =>
            `${Number(value).toFixed(1)}%`
          }
        />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
