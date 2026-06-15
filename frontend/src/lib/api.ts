const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Detection {
  class: string;
  class_id: number;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface InspectResponse {
  id: string;
  total_graos: number;
  class_counts: Record<string, number>;
  detections: Detection[];
  image_width: number;
  image_height: number;
}

export interface ExplainResponse {
  resposta: string;
  sugestoes: string[];
}

export async function inspectImage(imageBase64: string): Promise<InspectResponse> {
  // `imageBase64` já é o data URL lido do arquivo (uma única leitura, feita no
  // componente). O backend remove o prefixo "data:...;base64," se houver.
  let res: Response;
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 90_000); // cold start do HF Space
    try {
      res = await fetch(`${API_URL}/inspect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageBase64, imagem_url: "" }),
        signal: ctrl.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error(
        "O servidor de visão demorou demais para responder (pode estar acordando). Tente de novo em alguns segundos."
      );
    }
    const host = hostOf(API_URL);
    throw new Error(
      `Não consegui falar com o servidor de visão${host ? ` (${host})` : ""}. Verifique se ele está no ar.`
    );
  }

  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(err.detail ?? `Erro do servidor de visão (HTTP ${res.status}).`);
  }
  return res.json() as Promise<InspectResponse>;
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}


export interface ChatMsg {
  role: "user" | "assistant";
  content: string;
}

// Chat com a Vígil.ia em streaming: chama `onToken` a cada pedaço de texto
// e resolve com a resposta completa. /api/explain roda como API route do Next
// (Groq · Llama 3.3) na mesma origem da Vercel.
export async function explainClassStream(
  classe: string,
  pergunta: string,
  modo: "academico" | "industrial",
  historico: ChatMsg[],
  onToken: (chunk: string) => void
): Promise<string> {
  const res = await fetch(`/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ classe, pergunta, modo, historico }),
  });
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    if (chunk) {
      full += chunk;
      onToken(chunk);
    }
  }
  return full;
}
