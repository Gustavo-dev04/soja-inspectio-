"use client";
import { useState } from "react";
import { explainClass, type ExplainResponse } from "@/lib/api";
import { CLASS_LABELS } from "@/components/DefectTable";
import InspectionLogo from "@/components/InspectionLogo";

interface Props {
  classe: string;
  modo: "academico" | "industrial";
}

// Perguntas iniciais (a IA é OPCIONAL — só roda quando o usuário clica numa).
function starterQuestions(label: string, modo: Props["modo"]): string[] {
  const l = label.toLowerCase();
  if (modo === "industrial") {
    return [
      `O que significa "${l}" no lote?`,
      "Impacto na classificação comercial",
      "Que ação operacional tomar?",
    ];
  }
  return [
    `O que caracteriza ${l}?`,
    "Quais as causas agronômicas?",
    "Como reduzir na lavoura?",
  ];
}

export default function ExplainPanel({ classe, modo }: Props) {
  const [data, setData] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [perguntaAtiva, setPerguntaAtiva] = useState<string | undefined>(undefined);

  const label = CLASS_LABELS[classe] ?? classe;
  const starters = starterQuestions(label, modo);

  async function ask(pergunta: string) {
    setLoading(true);
    setPerguntaAtiva(pergunta);
    try {
      const result = await explainClass(classe, pergunta, modo);
      setData(result);
    } catch {
      setData({
        resposta:
          "Não foi possível obter a explicação agora. Tente novamente em instantes.",
        sugestoes: [],
      });
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setData(null);
    setPerguntaAtiva(undefined);
  }

  const chip = (q: string, active: boolean) =>
    `text-xs px-3 py-1.5 rounded-full border transition-colors disabled:opacity-50 ${
      active
        ? "bg-brand text-neutral-950 border-brand"
        : "border-white/15 text-neutral-300 hover:border-brand hover:text-brand"
    }`;

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-medium text-neutral-100">
            Pergunte à Vígil<span className="text-brand">.ia</span> sobre {label}
          </h4>
          <span className="rounded-full border border-white/10 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-neutral-500">
            opcional
          </span>
        </div>
        {data && !loading && (
          <button
            onClick={reset}
            className="text-xs text-neutral-500 underline-offset-2 hover:text-neutral-300 hover:underline"
          >
            limpar
          </button>
        )}
      </div>

      {/* estado: gerando — símbolo girando */}
      {loading ? (
        <div className="flex flex-col items-center gap-3 py-8 text-center">
          <InspectionLogo inspecting className="w-12 text-neutral-200" />
          <p className="animate-pulse text-sm text-neutral-400">
            Gerando resposta…
          </p>
          {perguntaAtiva && (
            <p className="max-w-sm text-xs text-neutral-600">“{perguntaAtiva}”</p>
          )}
        </div>
      ) : data ? (
        /* estado: resposta pronta */
        <div className="mt-4 space-y-4">
          {perguntaAtiva && (
            <p className="text-sm font-medium text-neutral-300">“{perguntaAtiva}”</p>
          )}
          <p className="whitespace-pre-line text-sm leading-relaxed text-neutral-200">
            {data.resposta}
          </p>

          {data.sugestoes.length > 0 && (
            <div className="space-y-2 pt-1">
              <p className="text-[11px] uppercase tracking-wider text-neutral-500">
                Continuar perguntando
              </p>
              <div className="flex flex-wrap gap-2">
                {data.sugestoes.map((s) => (
                  <button key={s} onClick={() => ask(s)} className={chip(s, false)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        /* estado inicial: perguntinhas (não chama a IA até clicar) */
        <div className="mt-4 space-y-3">
          <p className="text-xs text-neutral-500">
            Toque numa pergunta para a Vígil.ia gerar uma explicação.
          </p>
          <div className="flex flex-wrap gap-2">
            {starters.map((s) => (
              <button key={s} onClick={() => ask(s)} className={chip(s, false)}>
                {s}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
