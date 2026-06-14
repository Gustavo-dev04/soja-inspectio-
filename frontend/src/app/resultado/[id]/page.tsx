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

  if (error) return <p className="text-red-500 font-medium">{error}</p>;
  if (!data)
    return <p className="text-gray-400 animate-pulse">Carregando resultados...</p>;

  const uniqueClasses = Object.keys(data.class_counts);
  const defectClasses = uniqueClasses.filter((c) => c !== "intact");
  const hasDefects = defectClasses.length > 0;

  return (
    <div className="space-y-6">
      {/* Header + mode toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Resultado da Inspeção</h2>
          <p className="text-gray-400 text-xs font-mono mt-0.5">{id}</p>
        </div>

        <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1 gap-1 self-start sm:self-auto">
          <button
            onClick={() => setModo("academico")}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              modo === "academico"
                ? "bg-green-600 text-white shadow-sm"
                : "text-gray-500 hover:text-gray-800"
            }`}
          >
            Acadêmico
          </button>
          <button
            onClick={() => setModo("industrial")}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              modo === "industrial"
                ? "bg-gray-800 text-white shadow-sm"
                : "text-gray-500 hover:text-gray-800"
            }`}
          >
            Industrial
          </button>
        </div>
      </div>

      {/* Imagem + tabela */}
      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <h3 className="font-semibold text-gray-700 mb-3 text-sm uppercase tracking-wide">
            Imagem com Detecções
          </h3>
          {data.imageDataUrl ? (
            <BoundingBoxCanvas
              imageDataUrl={data.imageDataUrl}
              detections={data.detections}
              imageWidth={data.image_width}
              imageHeight={data.image_height}
            />
          ) : (
            <p className="text-gray-400 text-sm">Imagem não disponível.</p>
          )}
        </div>

        <div>
          <h3 className="font-semibold text-gray-700 mb-3 text-sm uppercase tracking-wide">
            {data.total_graos} grãos detectados
          </h3>
          <DefectTable classCounts={data.class_counts} totalGraos={data.total_graos} />

          {modo === "industrial" && hasDefects && (
            <div className="mt-4 p-3 rounded-lg bg-amber-50 border border-amber-200">
              <p className="text-xs font-semibold text-amber-800 uppercase tracking-wide mb-1">
                Resumo Industrial
              </p>
              <p className="text-sm text-amber-900">
                {defectClasses.length === 1
                  ? `Defeito predominante: ${CLASS_LABELS[defectClasses[0]] ?? defectClasses[0]}.`
                  : `${defectClasses.length} tipos de defeito detectados.`}{" "}
                {((defectClasses.reduce((s, c) => s + (data.class_counts[c] ?? 0), 0) / data.total_graos) * 100).toFixed(1)}% do lote
                apresenta algum defeito.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Painel de análise */}
      {uniqueClasses.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-700 text-sm uppercase tracking-wide">
              {modo === "academico" ? "Análise Técnica" : "Diagnóstico Operacional"}
            </h3>
            <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">
              Groq · Llama 3
            </span>
          </div>

          {/* Tabs por classe */}
          {uniqueClasses.length > 1 && (
            <div className="flex flex-wrap gap-2">
              {uniqueClasses.map((cls) => (
                <button
                  key={cls}
                  onClick={() => setClasseAtiva(cls)}
                  className={`text-sm px-3 py-1 rounded-full border transition-colors ${
                    classeAtiva === cls
                      ? "bg-green-600 text-white border-green-600"
                      : "border-gray-300 text-gray-600 hover:border-green-600 hover:text-green-700"
                  }`}
                >
                  {CLASS_LABELS[cls] ?? cls}
                  <span className="ml-1.5 text-xs opacity-70">{data.class_counts[cls]}</span>
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
