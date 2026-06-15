"use client";
import { useEffect, useRef, useState } from "react";
import { explainClass, type ChatMsg } from "@/lib/api";
import { CLASS_LABELS } from "@/components/DefectTable";
import InspectionLogo from "@/components/InspectionLogo";

interface Props {
  classe: string;
  modo: "academico" | "industrial";
}

// Sugestões iniciais (atalhos) — a partir daí é tudo digitando.
function starterQuestions(label: string, modo: Props["modo"]): string[] {
  const l = label.toLowerCase();
  if (modo === "industrial") {
    return [
      `O que significa "${l}" no lote?`,
      "Impacto na classificação comercial",
      "Que ação operacional tomar?",
    ];
  }
  return [`O que caracteriza ${l}?`, "Quais as causas agronômicas?", "Como reduzir na lavoura?"];
}

export default function ExplainPanel({ classe, modo }: Props) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const label = CLASS_LABELS[classe] ?? classe;
  const starters = starterQuestions(label, modo);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function send(text: string) {
    const q = text.trim();
    if (!q || loading) return;
    const historico = messages.slice(-8);
    setMessages((m) => [...m, { role: "user", content: q }]);
    setInput("");
    setLoading(true);
    try {
      const res = await explainClass(classe, q, modo, historico);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.resposta || "Não obtive uma resposta." },
      ]);
    } catch {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: "Não consegui responder agora. Tente de novo em instantes." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-white/10 bg-white/[0.02]">
      {/* cabeçalho */}
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-3">
        <InspectionLogo inspecting={loading} className="w-5 text-neutral-200" />
        <h4 className="text-sm font-medium text-neutral-100">
          Pergunte à Vígil<span className="text-brand">.ia</span> sobre {label}
        </h4>
        <span className="ml-auto rounded-full border border-white/10 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-neutral-500">
          Llama 3.3
        </span>
      </div>

      {/* thread */}
      <div ref={scrollRef} className="max-h-80 space-y-3 overflow-y-auto px-4 py-4">
        {messages.length === 0 && !loading && (
          <p className="text-xs text-neutral-500">
            Digite uma pergunta sobre {label.toLowerCase()} ou toque numa sugestão abaixo.
          </p>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-brand/15 px-3.5 py-2 text-sm text-neutral-100">
                {m.content}
              </div>
            </div>
          ) : (
            <div key={i} className="flex justify-start">
              <div className="max-w-[90%] whitespace-pre-line rounded-2xl rounded-bl-sm bg-white/[0.04] px-3.5 py-2 text-sm leading-relaxed text-neutral-200">
                {m.content}
              </div>
            </div>
          )
        )}

        {loading && (
          <div className="flex justify-start">
            <div className="flex items-center gap-1 rounded-2xl rounded-bl-sm bg-white/[0.04] px-4 py-3">
              <span className="typing-dot h-1.5 w-1.5 rounded-full bg-neutral-400" />
              <span className="typing-dot h-1.5 w-1.5 rounded-full bg-neutral-400" style={{ animationDelay: "0.15s" }} />
              <span className="typing-dot h-1.5 w-1.5 rounded-full bg-neutral-400" style={{ animationDelay: "0.3s" }} />
            </div>
          </div>
        )}
      </div>

      {/* sugestões (só com a conversa vazia) */}
      {messages.length === 0 && !loading && (
        <div className="flex flex-wrap gap-2 px-4 pb-3">
          {starters.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              className="rounded-full border border-white/15 px-3 py-1.5 text-xs text-neutral-300 transition-colors hover:border-brand hover:text-brand"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* barra de digitação */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex items-end gap-2 border-t border-white/10 p-3"
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          rows={1}
          placeholder={`Pergunte sobre ${label.toLowerCase()}…`}
          className="max-h-32 flex-1 resize-none rounded-xl border border-white/10 bg-white/[0.03] px-3.5 py-2.5 text-sm text-neutral-100 placeholder:text-neutral-600 focus:border-white/25 focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          aria-label="Enviar"
          className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-neutral-100 text-neutral-900 transition-opacity hover:bg-white disabled:opacity-30"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19V5" />
            <path d="m5 12 7-7 7 7" />
          </svg>
        </button>
      </form>
    </div>
  );
}
