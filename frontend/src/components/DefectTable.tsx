interface Props {
  classCounts: Record<string, number>;
  totalGraos: number;
}

export const CLASS_LABELS: Record<string, string> = {
  intact: "Intacto",
  immature: "Imaturo",
  broken: "Quebrado",
  "skin-damaged": "Casca danificada",
  spotted: "Manchado",
};

export const CLASS_COLORS: Record<string, string> = {
  intact: "#22c55e",
  immature: "#84cc16",
  broken: "#8b5cf6",
  "skin-damaged": "#f59e0b",
  spotted: "#ef4444",
};

export default function DefectTable({ classCounts, totalGraos }: Props) {
  const rows = Object.entries(classCounts).sort((a, b) => b[1] - a[1]);

  return (
    <div className="overflow-hidden rounded-xl border border-white/10 bg-white/[0.02]">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-wider text-neutral-400">
            <th className="px-4 py-2.5 font-medium">Classe</th>
            <th className="px-4 py-2.5 text-right font-medium">Qtd</th>
            <th className="px-4 py-2.5 text-right font-medium">%</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([cls, count]) => (
            <tr key={cls} className="border-t border-white/10 text-neutral-200">
              <td className="px-4 py-3">
                <span className="flex items-center gap-2.5">
                  <span
                    className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full"
                    style={{ backgroundColor: CLASS_COLORS[cls] ?? "#6b7280" }}
                  />
                  {CLASS_LABELS[cls] ?? cls}
                </span>
              </td>
              <td className="px-4 py-3 text-right tabular-nums">{count}</td>
              <td className="px-4 py-3 text-right tabular-nums text-neutral-400">
                {totalGraos > 0
                  ? ((count / totalGraos) * 100).toFixed(1) + "%"
                  : "—"}
              </td>
            </tr>
          ))}
          <tr className="border-t border-white/15 bg-white/[0.03] font-semibold text-neutral-100">
            <td className="px-4 py-3">Total</td>
            <td className="px-4 py-3 text-right tabular-nums">{totalGraos}</td>
            <td className="px-4 py-3 text-right tabular-nums">100%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
