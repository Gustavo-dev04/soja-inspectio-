import type { Metadata } from "next";
import InspectionLogo from "@/components/InspectionLogo";
import { CLASS_LABELS, CLASS_COLORS } from "@/components/DefectTable";

export const metadata: Metadata = {
  title: "Sobre — Vígil.ia",
  description: "O que é a Vígil.ia, como funciona e como ela evoluiu.",
};

const CLASS_ORDER = ["intact", "immature", "broken", "skin-damaged", "spotted"];

const METRICS = [
  { modelo: "EfficientNet-B0 — receita antiga", val: "64,0%", gap: "25,0%", atual: false },
  { modelo: "EfficientNet-B0 — em igualdade", val: "75,0%", gap: "7,2%", atual: false },
  { modelo: "YOLO11s-cls — produção", val: "91,7%", gap: "3,9%", atual: true },
];

const TIMELINE = [
  {
    tag: "Linhagem",
    titulo: "Saga v0 — EfficientNet-B0",
    texto:
      "Primeiro motor de inspeção (pipeline OpenCV → EfficientNet-B0, TensorFlow/Keras). 85% no dataset curado, mas 64% no domínio real — um gap de domain shift de −21 pontos. Nasce dentro da plataforma SPEXT como o modelo de grãos densos.",
  },
  {
    tag: "Migração",
    titulo: "Troca para YOLO11s-cls",
    texto:
      "Em experimento controlado (mesmas 57 fotos reais, mesmo recorte e split), o YOLO11s-cls bateu o EfficientNet no domínio real: 91,7% contra 75% em igualdade de condições. Ganhos atribuídos a AdamW + weight decay, EMA dos pesos e ao bloco de atenção espacial C2PSA do YOLO11.",
  },
  {
    tag: "Robustez",
    titulo: "Fine-tuning no domínio real (obrigatório)",
    texto:
      "O modelo treina em fundo preto; a foto do celular tem fundo cinza e luz variada. Sem adaptação, a acurácia despencava para ~3-8%. O fine-tuning no domínio real (freeze parcial + augmentation forte de brilho/cor) recupera para 91,7% — por isso deixou de ser opcional.",
  },
  {
    tag: "Atual · b1.0.0",
    titulo: "Vígil.ia — beta",
    texto:
      "Nova identidade e arquitetura de produto: modo 1 grão por foto, visão em Hugging Face Spaces (FastAPI + YOLO11s-cls), interface e análise em Next.js/Vercel, dados no Supabase e explicação agronômica opcional via Groq · Llama 3.3.",
    atual: true,
  },
];

const ROADMAP = [
  "Coletar 500 correções humanas (hoje 57/500) → fine-tuning supervisionado v1.",
  "Reativar o multi-grão (segmentação OpenCV) para inspeção de lote inteiro.",
  "Router de domínio leve (~2 ms) despachando grãos × superfícies sob uma API única.",
  "Tokenização semântica por imagem (horizonte v2+).",
];

const STACK = [
  ["Visão", "Ultralytics YOLO11s-cls (PyTorch) + OpenCV"],
  ["LLM (opcional)", "Groq · Llama 3.3 70B"],
  ["Interface", "Next.js · Vercel"],
  ["API de visão", "FastAPI · Hugging Face Spaces (Docker)"],
  ["Dados", "Supabase"],
  ["Dataset", "Roboflow “SoyaBeans Classifications v2” · MIT · ~12,5k imagens"],
];

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-4">
      <h2 className="text-[11px] font-medium uppercase tracking-wider text-neutral-400">
        {title}
      </h2>
      {children}
    </section>
  );
}

export default function SobrePage() {
  return (
    <div className="mx-auto max-w-3xl space-y-12 py-4">
      {/* cabeçalho */}
      <header className="flex flex-col items-center text-center">
        <InspectionLogo className="w-20 text-neutral-200" />
        <div className="mt-6 flex items-center gap-3">
          <h1 className="text-4xl font-light tracking-tight text-neutral-50">
            Vígil<span className="text-brand">.ia</span>
          </h1>
          <span className="rounded-full border border-brand/30 bg-brand/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.2em] text-brand">
            beta
          </span>
        </div>
        <p className="mt-3 max-w-md text-sm leading-relaxed text-neutral-500">
          Inspeção visual de grãos de soja por imagem. Você fotografa o grão e a
          Vígil.ia classifica a qualidade em uma de cinco classes.
        </p>
        <span className="mt-4 font-mono text-[10px] tracking-wider text-neutral-700">
          b1.0.0
        </span>
      </header>

      <Section title="As 5 classes">
        <div className="flex flex-wrap gap-2">
          {CLASS_ORDER.map((c) => (
            <span
              key={c}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.02] px-3 py-1.5 text-sm text-neutral-200"
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: CLASS_COLORS[c] }}
              />
              {CLASS_LABELS[c]}
            </span>
          ))}
        </div>
      </Section>

      <Section title="Como funciona">
        <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5 text-sm leading-relaxed text-neutral-300">
          <p className="font-mono text-xs text-neutral-400">
            foto → YOLO11s-cls → classe + confiança → resultado → análise opcional (Llama)
          </p>
          <p className="mt-3">
            O núcleo é um classificador <strong className="font-medium text-neutral-100">YOLO11s-cls</strong>{" "}
            treinado por transfer learning. Hoje a Vígil.ia opera em{" "}
            <strong className="font-medium text-neutral-100">modo 1 grão</strong>: a foto inteira é
            classificada como um único grão. O modo de lote (recorte de vários grãos com OpenCV)
            existe no código e entra no roadmap.
          </p>
        </div>
      </Section>

      <Section title="Desempenho — no domínio real">
        <div className="overflow-hidden rounded-xl border border-white/10 bg-white/[0.02]">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wider text-neutral-400">
                <th className="px-4 py-2.5 font-medium">Modelo</th>
                <th className="px-4 py-2.5 text-right font-medium">Acurácia real</th>
                <th className="px-4 py-2.5 text-right font-medium">Gap treino-val</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map((m) => (
                <tr
                  key={m.modelo}
                  className={`border-t border-white/10 ${
                    m.atual ? "text-neutral-100" : "text-neutral-300"
                  }`}
                >
                  <td className="px-4 py-3">
                    {m.modelo}
                    {m.atual && (
                      <span className="ml-2 rounded-full border border-brand/30 bg-brand/10 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-brand">
                        atual
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right font-medium tabular-nums">{m.val}</td>
                  <td className="px-4 py-3 text-right tabular-nums text-neutral-400">{m.gap}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs leading-relaxed text-neutral-500">
          Honestidade metodológica: a validação real tem apenas 12 fotos, então 91,7% equivale a
          11/12 — uma tendência forte, não uma métrica estatística. O único erro foi confundir
          “casca danificada” com “quebrado”, a fronteira mais ambígua do dataset.
        </p>
      </Section>

      <Section title="Linha do tempo">
        <ol className="relative space-y-6 border-l border-white/10 pl-6">
          {TIMELINE.map((t) => (
            <li key={t.titulo} className="relative">
              <span
                className={`absolute -left-[27px] top-1.5 h-2.5 w-2.5 rounded-full ring-4 ring-[#0a0a0b] ${
                  t.atual ? "bg-brand" : "bg-neutral-600"
                }`}
              />
              <p className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
                {t.tag}
              </p>
              <h3 className="mt-0.5 text-sm font-medium text-neutral-100">{t.titulo}</h3>
              <p className="mt-1 text-sm leading-relaxed text-neutral-400">{t.texto}</p>
            </li>
          ))}
        </ol>
      </Section>

      <Section title="Roadmap">
        <ul className="space-y-2">
          {ROADMAP.map((r) => (
            <li key={r} className="flex gap-3 text-sm text-neutral-300">
              <span className="mt-2 h-1 w-1 flex-shrink-0 rounded-full bg-brand" />
              {r}
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Stack">
        <dl className="overflow-hidden rounded-xl border border-white/10 bg-white/[0.02] divide-y divide-white/10">
          {STACK.map(([k, v]) => (
            <div key={k} className="flex flex-col gap-1 px-4 py-3 sm:flex-row sm:items-center sm:gap-4">
              <dt className="w-40 flex-shrink-0 text-[11px] uppercase tracking-wider text-neutral-500">
                {k}
              </dt>
              <dd className="text-sm text-neutral-200">{v}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <p className="pt-2 text-center font-mono text-[10px] tracking-wider text-neutral-700">
        Vígil.ia · b1.0.0 · 2026
      </p>
    </div>
  );
}
