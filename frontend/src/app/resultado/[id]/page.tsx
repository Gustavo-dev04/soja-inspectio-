"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
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
    let cached: string | null = null;
    try {
      cached = sessionStorage.getItem(`inspection_${id}`);
    } catch {
      /* storage indisponível — cai pro Supabase abaixo */
    }
    if (cached) {
      try {
        const parsed = JSON.parse(cached) as StoredResult;
        setData(parsed);
        const firstClass = Object.keys(parsed.class_counts)[0] ?? null;
        setClasseAtiva(firstClass);
        return;
      } catch {
        /* cache corrompido — segue pro Supabase */
      }
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

  // classe dominante (p/ o card de veredito) + confiança média dela
  const dominant = uniqueClasses
    .slice()
    .sort((a, b) => data.class_counts[b] - data.class_counts[a])[0];
  const domDets = data.detections.filter((d) => d.class === dominant);
  const domConf = domDets.length
    ? domDets.reduce((s, d) => s + d.confidence, 0) / domDets.length
    : 0;
  const single = data.total_graos === 1;

  // ---- FOCO DO NEGÓCIO: premium (intacto). Corte de confiança 80%. ----
  const PREMIUM_CONF = 0.8; // grão só é Premium se intacto E confiança >= 80%
  const LOTE_TARGET = 0.9; // lote aprovado se >= 90% premium
  const premiumCount = data.detections.filter(
    (d) => d.class === "intact" && d.confidence >= PREMIUM_CONF
  ).length;
  const premiumPct = data.total_graos ? premiumCount / data.total_graos : 0;
  const grainStatus: "premium" | "review" | "defect" =
    dominant === "intact" ? (domConf >= PREMIUM_CONF ? "premium" : "review") : "defect";
  const loteVerdict = single
    ? grainStatus === "premium"
      ? { txt: "APROVADO", sub: "grão premium — intacto", color: "#22c55e" }
      : grainStatus === "review"
      ? { txt: "REVISAR", sub: "intacto, mas confiança < 80%", color: "#f59e0b" }
      : {
          txt: "REPROVADO",
          sub: `fora do padrão — ${CLASS_LABELS[dominant] ?? dominant}`,
          color: CLASS_COLORS[dominant] ?? "#ef4444",
        }
    : premiumPct >= LOTE_TARGET
    ? {
        txt: "APROVADO",
        sub: `${(premiumPct * 100).toFixed(0)}% premium · meta ≥ ${LOTE_TARGET * 100}%`,
        color: "#22c55e",
      }
    : {
        txt: "REPROVADO",
        sub: `${(premiumPct * 100).toFixed(0)}% premium · meta ≥ ${LOTE_TARGET * 100}%`,
        color: "#ef4444",
      };

  return (
    <div className="space-y-7">
      {/* Nova inspeção + header + mode toggle */}
      <div className="reveal space-y-5">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-sm text-neutral-300 transition-colors hover:border-white/25 hover:text-neutral-100"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14" />
            <path d="M5 12h14" />
          </svg>
          Nova inspeção
        </Link>

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
      </div>

      {/* Card de veredito */}
      {dominant && (
        <ResultVerdict
          label={CLASS_LABELS[dominant] ?? dominant}
          color={CLASS_COLORS[dominant] ?? "#16a34a"}
          confidence={domConf}
          totalGraos={data.total_graos}
          single={single}
          status={grainStatus}
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

          {modo === "industrial" && (
            <div
              className="mt-4 rounded-xl border p-4"
              style={{
                borderColor: `${loteVerdict.color}40`,
                background: `${loteVerdict.color}12`,
              }}
            >
              <p
                className="text-[11px] font-medium uppercase tracking-wider"
                style={{ color: loteVerdict.color }}
              >
                Controle de qualidade · Premium
              </p>
              <p
                className="mt-1 text-2xl font-semibold tracking-tight"
                style={{ color: loteVerdict.color }}
              >
                {loteVerdict.txt}
              </p>
              <p className="mt-1 text-sm text-neutral-300">{loteVerdict.sub}</p>
              {!single && (
                <p className="mt-2 text-xs text-neutral-500">
                  {premiumCount} de {data.total_graos} grãos premium (intactos, confiança ≥ 80%)
                </p>
              )}
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
