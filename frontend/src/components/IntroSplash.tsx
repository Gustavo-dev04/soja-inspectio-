"use client";
import { useEffect, useState } from "react";
import InspectionLogo from "@/components/InspectionLogo";

/**
 * Abertura do site. Renderiza JÁ no primeiro paint (cobre tudo com fundo preto),
 * então não há flash do hero antes da animação. Toca por completo na 1ª vez da
 * sessão; em visitas seguintes some quase instantâneo.
 */
export default function IntroSplash() {
  const [leaving, setLeaving] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const seen = sessionStorage.getItem("vigil_intro");
    sessionStorage.setItem("vigil_intro", "1");
    const hold = seen ? 150 : reduce ? 500 : 2700;
    const t1 = setTimeout(() => setLeaving(true), hold);
    const t2 = setTimeout(() => setDone(true), hold + 650);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);

  if (done) return null;

  return (
    <div
      className={`fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#0a0a0b] transition-opacity duration-700 ${
        leaving ? "pointer-events-none opacity-0" : "opacity-100"
      }`}
    >
      <InspectionLogo intro className="w-36 text-neutral-200 sm:w-40" />
      <div className="mt-9 overflow-hidden">
        <h1 className="intro-word text-5xl font-light tracking-tight text-neutral-50">Vígil</h1>
      </div>
      <span className="intro-word-2 mt-4 font-mono text-[11px] uppercase tracking-[0.3em] text-neutral-500">
        b1.0.0 · beta
      </span>
    </div>
  );
}
