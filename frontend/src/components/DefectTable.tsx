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
    <table className="w-full text-sm border-collapse">
      <thead>
        <tr className="bg-gray-100 text-left">
          <th className="p-3 font-semibold">Classe</th>
          <th className="p-3 font-semibold text-right">Qtd</th>
          <th className="p-3 font-semibold text-right">%</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([cls, count]) => (
          <tr key={cls} className="border-t">
            <td className="p-3 flex items-center gap-2">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: CLASS_COLORS[cls] ?? "#6b7280" }}
              />
              {CLASS_LABELS[cls] ?? cls}
            </td>
            <td className="p-3 text-right">{count}</td>
            <td className="p-3 text-right">
              {totalGraos > 0
                ? ((count / totalGraos) * 100).toFixed(1) + "%"
                : "—"}
            </td>
          </tr>
        ))}
        <tr className="border-t font-bold bg-gray-50">
          <td className="p-3">Total</td>
          <td className="p-3 text-right">{totalGraos}</td>
          <td className="p-3 text-right">100%</td>
        </tr>
      </tbody>
    </table>
  );
}
