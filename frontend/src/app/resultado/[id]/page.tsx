"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { supabase } from "@/lib/supabase";
import BoundingBoxCanvas from "@/components/BoundingBoxCanvas";
import DefectTable, { CLASS_LABELS, CLASS_COLORS } from "@/components/DefectTable";
import ExplainPanel from "@/components/ExplainPanel";
import ResultVerdict from "@/components/ResultVerdict";
import type { InspectResponse } from "@/lib/api";

type Modo = "academico" | "industrial";

interface StoredResult extends InspectResponse {
  imageDataUrl: string;
}

export default function ResultadoPage() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<StoredResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modo, setModo] = useState<Modo>("academico");
  const [classeAtiva, setClasseAtiva] = useState<string | null>(null);

  useEffect(() => {
    const cached = sessionStorage.getItem(`inspection_${id}`);
    if (cached) {
      const parsed = JSON.parse(cached) as StoredResult;
      setData(parsed);
      const firstClass = Object.keys(parsed.class_counts)[0] ?? null;
      setClasseAtiva(firstClass);
      return;
    }

    supabase
      .from("inspecoes")
      .select("*")
      .eq("id", id)
      .single()
      .then(({ data: row, error: err }) => {
        if (err || !row) {
          setError("Inspeção não encontrada.");
          return;
        }
        const r = row.resultado_json as InspectResponse;
        const parsed = { ...r, id: row.id, imageDataUrl: row.imagem_url ?? "" };
        setData(parsed);
        const firstClass = Object.keys(r.class_counts)[0] ?? null;
        setClasseAtiva(firstClass);
      });
  }, [id]);

  if (error) return <p className="font-medium text-red-400">{error}</p>;
  if (!data)
    return <p className="animate-pulse text-neutral-400">Carregando resultados…</p>;

  const uniqueClasses = Object.keys(data.class_counts);
  const defectClasses = uniqueClasses.filter((c) => c !== "intact");
  const hasDefects = defectClasses.length > 0;

  // classe dominante (p/ o card de veredito) + confiança média dela
  const dominant = uniqueClasses
    .slice()
    .sort((a, b) => data.class_counts[b] - data.class_counts[a])[0];
  const domDets = data.detections.filter((d) => d.class === dominant);
  const domConf = domDets.length
    ? domDets.reduce((s, d) => s + d.confidence, 0) / domDets.length
    : 0;
  const single = data.total_graos === 1;

  return (
    <div className="space-y-7">
      {/* Header + mode toggle */}
      <div className="reveal flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h2 className="text-2xl font-medium tracking-tight text-neutral-100">
            Resultado da inspeção
          </h2>
          <p className="mt-0.5 font-mono text-xs text-neutral-600">{id}</p>
        </div>

        <div className="inline-flex gap-1 self-start rounded-lg border border-white/10 bg-white/[0.03] p-1 sm:self-auto">
          <button
            onClick={() => setModo("academico")}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
              modo === "academico"
                ? "bg-brand text-neutral-950"
                : "text-neutral-400 hover:text-neutral-100"
            }`}
          >
            Acadêmico
          </button>
          <button
            onClick={() => setModo("industrial")}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
              modo === "industrial"
                ? "bg-neutral-100 text-neutral-900"
                : "text-neutral-400 hover:text-neutral-100"
            }`}
          >
            Industrial
          </button>
        </div>
      </div>

      {/* Card de veredito */}
      {dominant && (
        <ResultVerdict
          label={CLASS_LABELS[dominant] ?? dominant}
          color={CLASS_COLORS[dominant] ?? "#16a34a"}
          confidence={domConf}
          totalGraos={data.total_graos}
          single={single}
        />
      )}

      {/* Imagem + tabela */}
      <div className="grid gap-6 md:grid-cols-2">
        <div className="reveal" style={{ animationDelay: "80ms" }}>
          <h3 className="mb-3 text-[11px] font-medium uppercase tracking-wider text-neutral-400">
            Imagem com detecções
          </h3>
          {data.imageDataUrl ? (
            <div
              className="overflow-hidden rounded-xl ring-1 ring-white/10"
              style={{ boxShadow: `0 0 60px -20px ${CLASS_COLORS[dominant] ?? "#16a34a"}55` }}
            >
              <BoundingBoxCanvas
                imageDataUrl={data.imageDataUrl}
                detections={data.detections}
                imageWidth={data.image_width}
                imageHeight={data.image_height}
              />
            </div>
          ) : (
            <p className="text-sm text-neutral-500">Imagem não disponível.</p>
          )}
        </div>

        <div className="reveal" style={{ animationDelay: "140ms" }}>
          <h3 className="mb-3 text-[11px] font-medium uppercase tracking-wider text-neutral-400">
            {data.total_graos === 1
              ? "1 grão detectado"
              : `${data.total_graos} grãos detectados`}
          </h3>
          <DefectTable classCounts={data.class_counts} totalGraos={data.total_graos} />

          {modo === "industrial" && hasDefects && (
            <div className="mt-4 rounded-lg border border-amber-500/25 bg-amber-500/10 p-3">
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-amber-300">
                Resumo industrial
              </p>
              <p className="text-sm text-amber-100/90">
                {defectClasses.length === 1
                  ? `Defeito predominante: ${CLASS_LABELS[defectClasses[0]] ?? defectClasses[0]}.`
                  : `${defectClasses.length} tipos de defeito detectados.`}{" "}
                {(
                  (defectClasses.reduce((s, c) => s + (data.class_counts[c] ?? 0), 0) /
                    data.total_graos) *
                  100
                ).toFixed(1)}
                % do lote apresenta algum defeito.
              </p>
            </div>
          )}

          {/* Análise da IA (opcional) — logo abaixo do resultado das classes */}
          {uniqueClasses.length > 0 && (
            <div className="mt-6 space-y-4">
              <div className="flex items-center gap-2">
                <h3 className="text-[11px] font-medium uppercase tracking-wider text-neutral-400">
                  {modo === "academico" ? "Análise técnica" : "Diagnóstico operacional"}
                </h3>
                <span className="rounded-full border border-white/10 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-neutral-500">
                  Groq · Llama 3.3
                </span>
              </div>

              {/* Tabs por classe (só quando há mais de uma) */}
              {uniqueClasses.length > 1 && (
                <div className="flex flex-wrap gap-2">
                  {uniqueClasses.map((cls) => (
                    <button
                      key={cls}
                      onClick={() => setClasseAtiva(cls)}
                      className={`rounded-full border px-3 py-1 text-sm transition-colors ${
                        classeAtiva === cls
                          ? "border-brand bg-brand text-neutral-950"
                          : "border-white/15 text-neutral-300 hover:border-brand hover:text-brand"
                      }`}
                    >
                      {CLASS_LABELS[cls] ?? cls}
                      <span className="ml-1.5 text-xs opacity-70">
                        {data.class_counts[cls]}
                      </span>
                    </button>
                  ))}
                </div>
              )}

              {classeAtiva && (
                <ExplainPanel
                  key={`${classeAtiva}-${modo}`}
                  classe={classeAtiva}
                  modo={modo}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
