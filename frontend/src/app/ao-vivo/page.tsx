"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { warmupOnnx, classifyCanvas } from "@/lib/onnx";
import { grainRect } from "@/lib/segment";
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
  const cropRef = useRef<HTMLCanvasElement>(null); // 224x224 (entrada do modelo)
  const workRef = useRef<HTMLCanvasElement>(null); // canvas auxiliar p/ segmentação
  const detectedRef = useRef(false); // achou o grão (Otsu) neste quadro?
  const runningRef = useRef(false);
  const startingRef = useRef(false); // start() em andamento (entre clique e loop)
  const mountedRef = useRef(true);
  const inFlightRef = useRef(false);
  const rafRef = useRef<number>();

  const [detected, setDetected] = useState(false);

  const [phase, setPhase] = useState<Phase>("idle");
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ms, setMs] = useState(0);

  // Recorta o GRÃO (Otsu, igual ao treino) e desenha o quadrado central dele em
  // 224x224. Sem grão claro, cai no center-crop do quadro inteiro (fallback).
  const drawGrain224 = useCallback((): boolean => {
    const v = videoRef.current;
    const c = cropRef.current;
    const work = workRef.current;
    if (!v || !c || !work || !v.videoWidth) return false;
    const vw = v.videoWidth;
    const vh = v.videoHeight;
    const rect = grainRect(v, vw, vh, work);
    detectedRef.current = !!rect;
    let side: number;
    let sx: number;
    let sy: number;
    if (rect) {
      // quadrado central do recorte do grão (= resize+centercrop do treino)
      side = Math.min(rect.sw, rect.sh);
      sx = rect.sx + (rect.sw - side) / 2;
      sy = rect.sy + (rect.sh - side) / 2;
    } else {
      side = Math.min(vw, vh);
      sx = (vw - side) / 2;
      sy = (vh - side) / 2;
    }
    // willReadFrequently no PRIMEIRO getContext (classifyCanvas faz getImageData/frame)
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return false;
    ctx.drawImage(v, sx, sy, side, side, 0, 0, 224, 224);
    return true;
  }, []);

  const tick = useCallback(async () => {
    if (!runningRef.current) return;
    if (!inFlightRef.current && cropRef.current && drawGrain224()) {
      inFlightRef.current = true;
      const t0 = performance.now();
      try {
        const { cls, conf } = await classifyCanvas(cropRef.current);
        if (!runningRef.current || !mountedRef.current) return; // parou durante a inferência
        setMs(Math.round(performance.now() - t0));
        setDetected(detectedRef.current);
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
  }, [drawGrain224]);

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
        <canvas ref={workRef} className="hidden" />

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
            <div className="flex flex-col items-end gap-1">
              <span
                className="rounded-full border px-2.5 py-1 font-mono text-[10px]"
                style={{
                  color: detected ? "#34d399" : "#a3a3a3",
                  borderColor: detected ? "#34d39955" : "#ffffff26",
                  background: "rgba(0,0,0,0.4)",
                }}
              >
                {detected ? "● grão recortado" : "○ procurando grão…"}
              </span>
              <span className="rounded-full border border-white/15 bg-black/40 px-2.5 py-1 font-mono text-[10px] text-neutral-300">
                {ms} ms · ~{ms ? Math.round(1000 / ms) : 0} fps
              </span>
            </div>
          </div>
        )}

        {running && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div
              className="h-40 w-40 rounded-2xl border-2 transition-colors"
              style={{ borderColor: detected ? "#34d399b3" : "#ffffff4d" }}
            />
          </div>
        )}
      </div>

      <div className="flex items-center justify-between">
        <p className="max-w-md text-xs text-neutral-600">
          Inferência local (onnxruntime-web), com recorte do grão por Otsu — funciona
          melhor com <b className="text-neutral-400">fundo escuro e luz de cima</b> (a mira
          fica verde quando acha o grão). Vídeo ainda é experimental.
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
