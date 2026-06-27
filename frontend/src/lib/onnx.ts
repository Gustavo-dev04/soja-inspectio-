// Inferência do classificador YOLO11s-cls direto no navegador (onnxruntime-web).
// Sem servidor: a câmera roda em tempo real, local. O .onnx fica em /public/models.
//
// Ordem das classes = ordem dos índices do modelo (ImageFolder alfabético),
// validada contra o .pt: 0 broken, 1 immature, 2 intact, 3 skin-damaged, 4 spotted.
export const ONNX_CLASSES = [
  "broken",
  "immature",
  "intact",
  "skin-damaged",
  "spotted",
] as const;

const MODEL_URL = "/models/soja_yolo11s.onnx";
const ORT_VERSION = "1.20.1";

type Session = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ort: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  session: any;
};

let sessionPromise: Promise<Session> | null = null;

async function getSession(): Promise<Session> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import("onnxruntime-web");
      // single-thread (sem COOP/COEP) + wasm servido por CDN
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
      const session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      return { ort, session };
    })();
  }
  return sessionPromise;
}

/** Garante o modelo carregado (chamar uma vez, mostrando "carregando"). */
export async function warmupOnnx(): Promise<void> {
  await getSession();
}

/**
 * Classifica um canvas 224×224 (já com o grão centralizado / center-crop feito).
 * Pré-processamento idêntico ao do Ultralytics classify: RGB, planar CHW, /255,
 * sem normalização (mean=0,std=1). A saída do modelo já é softmax (head Classify).
 */
export async function classifyCanvas(
  canvas: HTMLCanvasElement
): Promise<{ idx: number; cls: string; conf: number; probs: Float32Array }> {
  const { ort, session } = await getSession();
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas 2D indisponível.");
  const { data } = ctx.getImageData(0, 0, 224, 224); // RGBA, [0..255]

  const N = 224 * 224;
  const input = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    input[i] = data[i * 4] / 255; // R
    input[N + i] = data[i * 4 + 1] / 255; // G
    input[2 * N + i] = data[i * 4 + 2] / 255; // B
  }

  const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);
  const feeds: Record<string, unknown> = {};
  feeds[session.inputNames[0]] = tensor;
  const out = await session.run(feeds);
  const probs = out[session.outputNames[0]].data as Float32Array;

  let idx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[idx]) idx = i;
  return { idx, cls: ONNX_CLASSES[idx] ?? String(idx), conf: probs[idx], probs };
}
