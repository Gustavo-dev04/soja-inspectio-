"use client";
import { useEffect, useRef, useState } from "react";

interface Props {
  label: string;        // rótulo PT da classe dominante
  color: string;        // cor da classe
  confidence: number;   // 0..1
  totalGraos: number;
  single: boolean;      // true = modo 1 grão
}

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export default function ResultVerdict({
  label,
  color,
  confidence,
  totalGraos,
  single,
}: Props) {
  const [shown, setShown] = useState(0);
  const raf = useRef<number>();

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setShown(confidence);
      return;
    }
    const start = performance.now();
    const dur = 950;
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / dur);
      setShown(confidence * easeOut(p));
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [confidence]);

  const R = 42;
  const C = 2 * Math.PI * R;
  const offset = C * (1 - shown);
  const pct = Math.round(shown * 100);
  const confTxt =
    confidence >= 0.85
      ? "alta confiança"
      : confidence >= 0.6
      ? "confiança média"
      : "baixa confiança — confirme";

  return (
    <div className="reveal relative overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02] p-6">
      {/* glow suave na cor da classe */}
      <div
        className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full blur-3xl"
        style={{ background: color, opacity: 0.13 }}
      />

      <div className="relative flex items-center gap-6">
        {/* anel de confiança */}
        <div className="relative h-28 w-28 flex-shrink-0">
          <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
            <circle cx="50" cy="50" r={R} fill="none" stroke="currentColor"
              strokeWidth="7" className="text-white/10" />
            <circle
              cx="50" cy="50" r={R} fill="none" stroke={color}
              strokeWidth="7" strokeLinecap="round"
              strokeDasharray={C} strokeDashoffset={offset}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-semibold tabular-nums text-neutral-50">
              {pct}
              <span className="text-sm text-neutral-400">%</span>
            </span>
            <span className="text-[9px] uppercase tracking-wider text-neutral-500">
              confiança
            </span>
          </div>
        </div>

        {/* classe dominante */}
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-wider text-neutral-400">
            {single ? "Classe identificada" : "Classe predominante"}
          </p>
          <div className="mt-1.5 flex items-center gap-2.5">
            <span className="h-3 w-3 flex-shrink-0 rounded-full" style={{ background: color }} />
            <h2 className="truncate text-2xl font-medium tracking-tight text-neutral-50">
              {label}
            </h2>
          </div>
          <p className="mt-1.5 text-sm text-neutral-500">
            {single ? "1 grão analisado" : `${totalGraos} grãos analisados`} · {confTxt}
          </p>
        </div>
      </div>
    </div>
  );
}
