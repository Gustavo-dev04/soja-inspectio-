"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { supabase } from "@/lib/supabase";
import BoundingBoxCanvas from "@/components/BoundingBoxCanvas";
import DefectTable from "@/components/DefectTable";
import type { InspectResponse } from "@/lib/api";

interface StoredResult extends InspectResponse {
  imageDataUrl: string;
}

export default function ResultadoPage() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<StoredResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const cached = sessionStorage.getItem(`inspection_${id}`);
    if (cached) {
      setData(JSON.parse(cached));
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
        setData({ ...r, id: row.id, imageDataUrl: row.imagem_url ?? "" });
      });
  }, [id]);

  if (error) return <p className="text-red-500 font-medium">{error}</p>;
  if (!data)
    return (
      <p className="text-gray-400 animate-pulse">Carregando resultados...</p>
    );

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 mb-1">
          Resultado da Inspeção
        </h2>
        <p className="text-gray-400 text-xs font-mono">{id}</p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <div>
          <h3 className="font-semibold text-gray-700 mb-3">
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
            <p className="text-gray-400 text-sm">
              Imagem não disponível para visualização direta.
            </p>
          )}
        </div>

        <div>
          <h3 className="font-semibold text-gray-700 mb-3">
            Resumo — {data.total_graos} grãos detectados
          </h3>
          <DefectTable
            classCounts={data.class_counts}
            totalGraos={data.total_graos}
          />
        </div>
      </div>
    </div>
  );
}
