import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

// LLM via API compatível com OpenAI. Default: Groq + Llama 3.3 70b (grátis, rápido).
const ENDPOINT =
  process.env.LLM_ENDPOINT ?? "https://api.groq.com/openai/v1/chat/completions";
const MODEL = process.env.LLM_MODEL ?? "llama-3.3-70b-versatile";

const CLASS_PT: Record<string, string> = {
  intact: "Intacto",
  immature: "Imaturo (verde)",
  broken: "Quebrado",
  "skin-damaged": "Casca danificada",
  spotted: "Manchado",
};

interface ChatMsg {
  role: "user" | "assistant";
  content: string;
}

interface ExplainBody {
  classe: string;
  pergunta?: string;
  modo?: "academico" | "industrial";
  historico?: ChatMsg[];
}

export async function POST(req: NextRequest) {
  const token = process.env.GROQ_API_KEY ?? process.env.LLM_API_KEY;
  if (!token) {
    return NextResponse.json({ detail: "GROQ_API_KEY não configurada" }, { status: 503 });
  }

  let body: ExplainBody;
  try {
    body = (await req.json()) as ExplainBody;
  } catch {
    return NextResponse.json({ detail: "JSON inválido" }, { status: 422 });
  }

  const { classe, pergunta, modo = "academico", historico = [] } = body;
  const classePt = CLASS_PT[classe] ?? classe;

  const modoInstrucao =
    modo === "academico"
      ? "Use linguagem técnico-científica. Mencione processos fisiológicos, impactos nutricionais e parâmetros de qualidade segundo normas do MAPA. Seja conciso: até 3 parágrafos."
      : "Seja direto e objetivo. Foque em impacto comercial, tolerâncias da Tabela de Classificação MAPA/CONAB e ações corretivas imediatas no campo ou silo. Até 3 parágrafos curtos.";

  const perguntaFinal =
    pergunta ??
    `Por que o grão de soja está classificado como '${classePt}' e quais são as principais causas?`;

  const system =
    "Você é a Vígil.ia, especialista em agronomia com foco em produção e qualidade de grãos de soja. " +
    `O grão em análise foi classificado como '${classePt}'. ` +
    "Responda sempre em português brasileiro, de forma conversacional, mantendo o contexto das mensagens anteriores. " +
    modoInstrucao;

  const hist = (Array.isArray(historico) ? historico : [])
    .filter((m) => m && (m.role === "user" || m.role === "assistant") && m.content)
    .slice(-8)
    .map((m) => ({ role: m.role, content: String(m.content).slice(0, 2000) }));

  let resp: Response;
  try {
    resp = await fetch(ENDPOINT, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [{ role: "system", content: system }, ...hist, { role: "user", content: perguntaFinal }],
        max_tokens: 512,
        temperature: 0.4,
        stream: true,
      }),
    });
  } catch {
    return NextResponse.json({ detail: "Falha ao contatar o provedor de LLM" }, { status: 502 });
  }

  if (!resp.ok || !resp.body) {
    const errText = await resp.text().catch(() => "");
    return NextResponse.json(
      { detail: `LLM retornou ${resp.status}: ${errText.slice(0, 200)}` },
      { status: 502 }
    );
  }

  // Reemite só o texto (deltas) do SSE da Groq como um stream de texto simples.
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      const encoder = new TextEncoder();
      let buf = "";
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() ?? "";
          for (const line of lines) {
            const t = line.trim();
            if (!t.startsWith("data:")) continue;
            const payload = t.slice(5).trim();
            if (payload === "[DONE]") {
              controller.close();
              return;
            }
            try {
              const json = JSON.parse(payload);
              const delta: string | undefined = json.choices?.[0]?.delta?.content;
              if (delta) controller.enqueue(encoder.encode(delta));
            } catch {
              /* ignora linhas parciais */
            }
          }
        }
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
    },
  });
}
