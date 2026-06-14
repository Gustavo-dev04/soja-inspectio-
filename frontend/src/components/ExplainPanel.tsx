"use client";
import { useState, useEffect, useCallback } from "react";
import { explainClass, type ExplainResponse } from "@/lib/api";
import { CLASS_LABELS } from "@/components/DefectTable";

interface Props {
  classe: string;
  modo: "academico" | "industrial";
}

export default function ExplainPanel({ classe, modo }: Props) {
  const [data, setData] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [perguntaAtiva, setPerguntaAtiva] = useState<string | undefined>(undefined);

  const fetchExplanation = useCallback(
    async (pergunta?: string) => {
      setLoading(true);
      try {
        const result = await explainClass(classe, pergunta, modo);
        setData(result);
        setPerguntaAtiva(pergunta);
      } catch {
        setData({
          resposta: "Não foi possível obter a explicação. Verifique se o backend está rodando.",
          sugestoes: [],
        });
      } finally {
        setLoading(false);
      }
    },
    [classe, modo]
  );

  useEffect(() => {
    setData(null);
    setPerguntaAtiva(undefined);
    fetchExplanation(undefined);
  }, [fetchExplanation]);

  const label = CLASS_LABELS[classe] ?? classe;

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-800 text-sm">
          {perguntaAtiva ? `"${perguntaAtiva}"` : `Análise: ${label}`}
        </h4>
        {perguntaAtiva && (
          <button
            onClick={() => fetchExplanation(undefined)}
            className="text-xs text-gray-400 hover:text-gray-600 underline"
          >
            voltar
          </button>
        )}
      </div>

      {loading ? (
        <div className="space-y-2 animate-pulse">
          <div className="h-3 bg-gray-100 rounded w-full" />
          <div className="h-3 bg-gray-100 rounded w-5/6" />
          <div className="h-3 bg-gray-100 rounded w-4/6" />
        </div>
      ) : data ? (
        <>
          <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-line">
            {data.resposta}
          </p>

          {data.sugestoes.length > 0 && (
            <div className="space-y-2 pt-1">
              <p className="text-xs text-gray-400 uppercase tracking-wider">
                Perguntas relacionadas
              </p>
              <div className="flex flex-wrap gap-2">
                {data.sugestoes.map((s) => (
                  <button
                    key={s}
                    onClick={() => fetchExplanation(s)}
                    disabled={loading || perguntaAtiva === s}
                    className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                      perguntaAtiva === s
                        ? "bg-green-600 text-white border-green-600"
                        : "border-gray-300 text-gray-600 hover:border-green-600 hover:text-green-700"
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}
