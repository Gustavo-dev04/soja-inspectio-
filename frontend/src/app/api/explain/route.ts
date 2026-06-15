import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

// LLM via API compatível com OpenAI. Default: Groq + Llama 3.3 70b (grátis, rápido).
// Trocável por env sem mexer no código.
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

const SUGESTOES: Record<string, string[]> = {
  immature: [
    "Como evitar grãos imaturos na colheita?",
    "Qual o impacto comercial do grão imaturo?",
    "Como identificar o ponto certo de colheita?",
  ],
  broken: [
    "Por que os grãos quebram durante a colheita?",
    "Como reduzir perdas por quebra?",
    "Qual % de quebra é tolerado pela CONAB?",
  ],
  "skin-damaged": [
    "O que causa danos à casca do grão?",
    "Como prevenir danos na colheita e transporte?",
    "Grão com casca danificada pode ser comercializado?",
  ],
  spotted: [
    "O que causa manchas nos grãos de soja?",
    "Manchas indicam contaminação fúngica?",
    "Como tratar grãos manchados no armazenamento?",
  ],
  intact: [
    "Como manter os grãos íntegros durante a colheita?",
    "Quais cuidados no armazenamento?",
    "Como a qualidade é avaliada no mercado?",
  ],
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
    return NextResponse.json(
      { detail: "GROQ_API_KEY não configurada" },
      { status: 503 }
    );
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

  // histórico recente (limitado) + a pergunta atual
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
        messages: [
          { role: "system", content: system },
          ...hist,
          { role: "user", content: perguntaFinal },
        ],
        max_tokens: 512,
        temperature: 0.4,
      }),
    });
  } catch {
    return NextResponse.json(
      { detail: "Falha ao contatar o provedor de LLM" },
      { status: 502 }
    );
  }

  if (!resp.ok) {
    const errText = await resp.text().catch(() => "");
    return NextResponse.json(
      { detail: `LLM retornou ${resp.status}: ${errText.slice(0, 200)}` },
      { status: 502 }
    );
  }

  const data = (await resp.json()) as {
    choices?: { message?: { content?: string } }[];
  };
  const resposta = data.choices?.[0]?.message?.content?.trim() ?? "";

  return NextResponse.json({
    resposta,
    sugestoes: SUGESTOES[classe] ?? [],
  });
}
