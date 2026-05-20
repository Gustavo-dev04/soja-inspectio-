import { createClient } from "@supabase/supabase-js";
import DashboardChart from "@/components/DashboardChart";

export const dynamic = "force-dynamic";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  { auth: { persistSession: false, autoRefreshToken: false } }
);

interface Inspecao {
  id: string;
  created_at: string;
  total_graos: number;
  resultado_json: { class_counts: Record<string, number> };
}

async function getStats() {
  const { data, error } = await supabase
    .from("inspecoes")
    .select("id, created_at, total_graos, resultado_json")
    .order("created_at", { ascending: false })
    .limit(500);

  if (error || !data) return { chartData: [], totalGraos: 0, totalInspecoes: 0 };

  const totals: Record<string, number> = {};
  let totalGraos = 0;

  for (const row of data as Inspecao[]) {
    for (const [cls, cnt] of Object.entries(
      row.resultado_json?.class_counts ?? {}
    )) {
      totals[cls] = (totals[cls] ?? 0) + cnt;
      totalGraos += cnt;
    }
  }

  const chartData = Object.entries(totals).map(([name, count]) => ({
    name,
    value: totalGraos > 0 ? (count / totalGraos) * 100 : 0,
  }));

  return { chartData, totalGraos, totalInspecoes: data.length };
}

export default async function DashboardPage() {
  const { chartData, totalGraos, totalInspecoes } = await getStats();

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold text-gray-800">Dashboard de Lotes</h2>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded-xl p-5 shadow-sm border">
          <p className="text-sm text-gray-500">Total de Inspeções</p>
          <p className="text-3xl font-bold text-brand">{totalInspecoes}</p>
        </div>
        <div className="bg-white rounded-xl p-5 shadow-sm border">
          <p className="text-sm text-gray-500">Total de Grãos Analisados</p>
          <p className="text-3xl font-bold text-brand">
            {totalGraos.toLocaleString("pt-BR")}
          </p>
        </div>
      </div>

      {chartData.length > 0 ? (
        <div className="bg-white rounded-xl p-6 shadow-sm border grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="font-semibold text-gray-700 mb-4">
              Distribuição por Classe (Pizza)
            </h3>
            <DashboardChart data={chartData} type="pie" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-4">
              Distribuição por Classe (Barras)
            </h3>
            <DashboardChart data={chartData} type="bar" />
          </div>
        </div>
      ) : (
        <p className="text-gray-400">
          Nenhuma inspeção encontrada. Envie imagens para começar.
        </p>
      )}
    </div>
  );
}
