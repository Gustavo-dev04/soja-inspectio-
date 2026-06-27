"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { inspectImage } from "@/lib/api";
import { CLASS_LABELS, CLASS_COLORS } from "@/components/DefectTable";

const PREMIUM_CONF = 0.8; // grão só é Premium se intacto E confiança >= 80%

type Status = "premium" | "review" | "defect";
type Verdict = { status: Status; label: string; color: string; conf: number };

const STATUS_TXT: Record<Status, string> = {
  premium: "PREMIUM",
  review: "REVISAR",
  defect: "FORA DO PADRÃO",
};

export default function AoVivoPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const runningRef = useRef(false);
  const warmRef = useRef(false);

  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<"idle" | "warming" | "live" | "error">("idle");
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState(0);

  const grab = useCallback((): string | null => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c || !v.videoWidth) return null;
    const maxDim = 384;
    const scale = Math.min(1, maxDim / Math.max(v.videoWidth, v.videoHeight));
    const w = Math.round(v.videoWidth * scale);
    const h = Math.round(v.videoHeight * scale);
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(v, 0, 0, w, h);
    return c.toDataURL("image/jpeg", 0.7);
  }, []);

  const loop = useCallback(async () => {
    if (!runningRef.current) return;
    const frame = grab();
    if (!frame) {
      setTimeout(loop, 200);
      return;
    }
    const t0 = performance.now();
    try {
      const r = await inspectImage(frame, {
        persist: false,
        timeoutMs: warmRef.current ? 15000 : 90000,
      });
      warmRef.current = true;
      setLatency(Math.round(performance.now() - t0));
      const d = r.detections[0];
      if (d) {
        const isIntact = d.class === "intact";
        const status: Status = isIntact
          ? d.confidence >= PREMIUM_CONF
            ? "premium"
            : "review"
          : "defect";
        setVerdict({
          status,
          label: CLASS_LABELS[d.class] ?? d.class,
          color:
            status === "premium"
              ? "#22c55e"
              : status === "review"
              ? "#f59e0b"
              : CLASS_COLORS[d.class] ?? "#ef4444",
          conf: d.confidence,
        });
      }
      setPhase("live");
    } catch {
      // não derruba o loop por causa de um frame que falhou
    }
    if (runningRef.current) setTimeout(loop, 120);
  }, [grab]);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      runningRef.current = true;
      warmRef.current = false;
      setRunning(true);
      setPhase("warming");
      loop();
    } catch {
      setPhase("error");
      setError(
        "Não consegui acessar a câmera. Permita o acesso no navegador (e use HTTPS)."
      );
    }
  }, [loop]);

  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    setPhase("idle");
    setVerdict(null);
    const v = videoRef.current;
    const s = v?.srcObject as MediaStream | null;
    s?.getTracks().forEach((t) => t.stop());
    if (v) v.srcObject = null;
  }, []);

  useEffect(() => () => stop(), [stop]);

  return (
    <div className="reveal space-y-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-medium tracking-tight text-neutral-100">
            Inspeção ao vivo
          </h2>
          <p className="mt-0.5 text-sm text-neutral-500">
            Aponte a câmera para um grão — o veredito premium atualiza em tempo real.
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
        <video
          ref={videoRef}
          playsInline
          muted
          className="aspect-[4/3] w-full object-cover"
        />
        <canvas ref={canvasRef} className="hidden" />

        {/* overlay de estado / veredito */}
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

        {phase === "warming" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <p className="animate-pulse text-sm text-neutral-200">
              Aquecendo o modelo… (primeira leitura pode demorar alguns segundos)
            </p>
          </div>
        )}

        {phase === "error" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 p-6 text-center">
            <p className="text-sm text-red-300">{error}</p>
          </div>
        )}

        {/* faixa de veredito ao vivo */}
        {phase === "live" && verdict && (
          <div
            className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-3 p-4"
            style={{
              background: `linear-gradient(to top, ${verdict.color}33, transparent)`,
            }}
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
              {latency} ms/quadro
            </span>
          </div>
        )}

        {/* mira central */}
        {running && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div className="h-40 w-40 rounded-2xl border-2 border-white/30" />
          </div>
        )}
      </div>

      <div className="flex items-center justify-between">
        <p className="text-xs text-neutral-600">
          Inferência no servidor (CPU) → ~2–3 quadros/s. Encha a mira com 1 grão,
          fundo escuro e luz de cima.
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
