"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { warmupOnnx, classifyCanvas } from "@/lib/onnx";
import { CLASS_LABELS, CLASS_COLORS } from "@/components/DefectTable";

const PREMIUM_CONF = 0.8; // grão só é Premium se intacto E confiança >= 80%

type Status = "premium" | "review" | "defect";
type Verdict = { status: Status; label: string; color: string; conf: number };
type Phase = "idle" | "loading" | "live" | "error";

const STATUS_TXT: Record<Status, string> = {
  premium: "PREMIUM",
  review: "REVISAR",
  defect: "FORA DO PADRÃO",
};

export default function AoVivoPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cropRef = useRef<HTMLCanvasElement>(null); // 224x224 (center-crop)
  const runningRef = useRef(false);
  const startingRef = useRef(false); // start() em andamento (entre clique e loop)
  const mountedRef = useRef(true);
  const inFlightRef = useRef(false);
  const rafRef = useRef<number>();

  const [phase, setPhase] = useState<Phase>("idle");
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ms, setMs] = useState(0);

  const drawCenter224 = useCallback((): boolean => {
    const v = videoRef.current;
    const c = cropRef.current;
    if (!v || !c || !v.videoWidth) return false;
    const side = Math.min(v.videoWidth, v.videoHeight);
    const sx = (v.videoWidth - side) / 2;
    const sy = (v.videoHeight - side) / 2;
    // willReadFrequently no PRIMEIRO getContext do canvas (o classifyCanvas faz
    // getImageData todo frame) — evita readback GPU->CPU por quadro.
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return false;
    ctx.drawImage(v, sx, sy, side, side, 0, 0, 224, 224); // center-crop + resize
    return true;
  }, []);

  const tick = useCallback(async () => {
    if (!runningRef.current) return;
    if (!inFlightRef.current && cropRef.current && drawCenter224()) {
      inFlightRef.current = true;
      const t0 = performance.now();
      try {
        const { cls, conf } = await classifyCanvas(cropRef.current);
        if (!runningRef.current || !mountedRef.current) return; // parou durante a inferência
        setMs(Math.round(performance.now() - t0));
        const isIntact = cls === "intact";
        const status: Status = isIntact
          ? conf >= PREMIUM_CONF
            ? "premium"
            : "review"
          : "defect";
        setVerdict({
          status,
          label: CLASS_LABELS[cls] ?? cls,
          color:
            status === "premium"
              ? "#22c55e"
              : status === "review"
              ? "#f59e0b"
              : CLASS_COLORS[cls] ?? "#ef4444",
          conf,
        });
        setPhase("live");
      } catch {
        /* ignora um frame que falhou */
      } finally {
        inFlightRef.current = false;
      }
    }
    if (runningRef.current) rafRef.current = requestAnimationFrame(tick);
  }, [drawCenter224]);

  const start = useCallback(async () => {
    if (runningRef.current || startingRef.current) return; // evita duplo-clique
    startingRef.current = true;
    setError(null);
    setPhase("loading");
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
        audio: false,
      });
      if (!startingRef.current || !mountedRef.current) {
        stream.getTracks().forEach((t) => t.stop()); // cancelado durante o getUserMedia
        return;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      await warmupOnnx(); // baixa/compila o modelo (≈21 MB na 1ª vez)
      if (!startingRef.current || !mountedRef.current) {
        stream.getTracks().forEach((t) => t.stop()); // cancelado durante o warmup
        if (videoRef.current) videoRef.current.srcObject = null;
        return;
      }
      runningRef.current = true;
      startingRef.current = false;
      tick();
    } catch (e) {
      startingRef.current = false;
      stream?.getTracks().forEach((t) => t.stop());
      if (!mountedRef.current) return;
      setPhase("error");
      setError(
        e instanceof Error && e.name === "NotAllowedError"
          ? "Permissão de câmera negada. Habilite o acesso e tente de novo."
          : "Não consegui iniciar (câmera ou modelo). Use HTTPS e um navegador atual."
      );
    }
  }, [tick]);

  const stop = useCallback(() => {
    runningRef.current = false;
    startingRef.current = false; // cancela um start() em andamento
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (mountedRef.current) {
      setPhase("idle");
      setVerdict(null);
    }
    const v = videoRef.current;
    const s = v?.srcObject as MediaStream | null;
    s?.getTracks().forEach((t) => t.stop());
    if (v) v.srcObject = null;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stop();
    };
  }, [stop]);

  const running = phase === "live" || phase === "loading";

  return (
    <div className="reveal space-y-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-medium tracking-tight text-neutral-100">
            Inspeção ao vivo
          </h2>
          <p className="mt-0.5 text-sm text-neutral-500">
            O modelo roda <b className="text-neutral-300">no seu navegador</b> — tempo
            real, sem servidor. Aponte para um grão.
          </p>
        </div>
        <Link
          href="/"
          className="rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-sm text-neutral-300 transition-colors hover:border-white/25 hover:text-neutral-100"
        >
          ← Voltar
        </Link>
      </div>

      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black">
        <video ref={videoRef} playsInline muted className="aspect-[4/3] w-full object-cover" />
        <canvas ref={cropRef} width={224} height={224} className="hidden" />

        {phase === "idle" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-black/40 text-center">
            <p className="max-w-xs text-sm text-neutral-300">
              Câmera desligada. Toque para começar a inspeção ao vivo.
            </p>
            <button
              onClick={start}
              className="rounded-full bg-neutral-100 px-6 py-2.5 text-sm font-medium text-neutral-900 transition-colors hover:bg-white"
            >
              Ligar câmera
            </button>
          </div>
        )}

        {phase === "loading" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 p-6 text-center">
            <p className="animate-pulse text-sm text-neutral-200">
              Carregando o modelo no navegador (≈21 MB, só na 1ª vez)…
            </p>
          </div>
        )}

        {phase === "error" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 p-6 text-center">
            <p className="text-sm text-red-300">{error}</p>
          </div>
        )}

        {phase === "live" && verdict && (
          <div
            className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-3 p-4"
            style={{ background: `linear-gradient(to top, ${verdict.color}33, transparent)` }}
          >
            <div>
              <p
                className="text-2xl font-semibold tracking-tight"
                style={{ color: verdict.color }}
              >
                {STATUS_TXT[verdict.status]}
              </p>
              <p className="text-sm text-neutral-200">
                {verdict.label} · {(verdict.conf * 100).toFixed(0)}% de confiança
              </p>
            </div>
            <span className="rounded-full border border-white/15 bg-black/40 px-2.5 py-1 font-mono text-[10px] text-neutral-300">
              {ms} ms · ~{ms ? Math.round(1000 / ms) : 0} fps
            </span>
          </div>
        )}

        {running && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div className="h-40 w-40 rounded-2xl border-2 border-white/30" />
          </div>
        )}
      </div>

      <div className="flex items-center justify-between">
        <p className="text-xs text-neutral-600">
          Inferência local (onnxruntime-web). Encha a mira com 1 grão, fundo escuro e luz de cima.
        </p>
        {running && (
          <button
            onClick={stop}
            className="rounded-full border border-white/15 px-4 py-1.5 text-sm text-neutral-300 transition-colors hover:border-red-400/40 hover:text-red-300"
          >
            Parar
          </button>
        )}
      </div>
    </div>
  );
}
