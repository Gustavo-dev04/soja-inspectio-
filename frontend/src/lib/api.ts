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

export async function inspectImage(file: File): Promise<InspectResponse> {
  const b64 = await fileToBase64(file);
  const res = await fetch(`${API_URL}/inspect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: b64, imagem_url: "" }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<InspectResponse>;
}

export async function explainClass(
  classe: string,
  pergunta?: string,
  modo: "academico" | "industrial" = "academico"
): Promise<ExplainResponse> {
  // /explain roda como API route do Next.js (Phi-4-mini via GitHub Models),
  // na mesma origem da Vercel — não usa NEXT_PUBLIC_API_URL (esse é só pro /inspect/YOLO).
  const res = await fetch(`/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ classe, pergunta, modo }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<ExplainResponse>;
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
