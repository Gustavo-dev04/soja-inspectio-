"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { supabase } from "@/lib/supabase";
import BoundingBoxCanvas from "@/components/BoundingBoxCanvas";
import DefectTable, { CLASS_LABELS } from "@/components/DefectTable";
import ExplainPanel from "@/components/ExplainPanel";
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

  return (
    <div className="space-y-7">
      {/* Header + mode toggle */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
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

      {/* Imagem + tabela */}
      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <h3 className="mb-3 text-[11px] font-medium uppercase tracking-wider text-neutral-400">
            Imagem com detecções
          </h3>
          {data.imageDataUrl ? (
            <BoundingBoxCanvas
              imageDataUrl={data.imageDataUrl}
              detections={data.detections}
              imageWidth={data.image_width}
              imageHeight={data.image_height}
            />
          ) : (
            <p className="text-sm text-neutral-500">Imagem não disponível.</p>
          )}
        </div>

        <div>
          <h3 className="mb-3 text-[11px] font-medium uppercase tracking-wider text-neutral-400">
            {data.total_graos} grãos detectados
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
        </div>
      </div>

      {/* Painel de análise (IA opcional) */}
      {uniqueClasses.length > 0 && (
        <div className="space-y-4 border-t border-white/5 pt-6">
          <div className="flex items-center gap-2">
            <h3 className="text-[11px] font-medium uppercase tracking-wider text-neutral-400">
              {modo === "academico" ? "Análise técnica" : "Diagnóstico operacional"}
            </h3>
            <span className="rounded-full border border-white/10 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-neutral-500">
              Groq · Llama 3.3
            </span>
          </div>

          {/* Tabs por classe */}
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
            <ExplainPanel key={`${classeAtiva}-${modo}`} classe={classeAtiva} modo={modo} />
          )}
        </div>
      )}
    </div>
  );
}
