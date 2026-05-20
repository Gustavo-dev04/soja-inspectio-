interface Props {
  classCounts: Record<string, number>;
  totalGraos: number;
}

const LABELS: Record<string, string> = {
  soja_boa: "Soja Boa",
  soja_verde: "Soja Verde (Imaturo)",
  soja_meia_lua: "Soja Meia-Lua",
  soja_ardida: "Soja Ardida",
  soja_quebrada: "Soja Quebrada",
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
            <td className="p-3">{LABELS[cls] ?? cls}</td>
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
