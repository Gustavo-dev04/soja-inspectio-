"use client";
import { useEffect, useState } from "react";
import InspectionLogo from "@/components/InspectionLogo";

type Phase = "hidden" | "playing" | "leaving" | "done";

/**
 * Abertura do site: mostra o símbolo se desenhando + a marca "Vígil" e some,
 * revelando o hero. Toca uma vez por sessão (sessionStorage).
 */
export default function IntroSplash() {
  const [phase, setPhase] = useState<Phase>("hidden");

  useEffect(() => {
    if (sessionStorage.getItem("vigil_intro")) return; // já viu nesta sessão
    sessionStorage.setItem("vigil_intro", "1");
    setPhase("playing");

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const hold = reduce ? 500 : 2700;
    const t1 = setTimeout(() => setPhase("leaving"), hold);
    const t2 = setTimeout(() => setPhase("done"), hold + 700);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);

  if (phase === "hidden" || phase === "done") return null;

  return (
    <div
      className={`fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#0a0a0b] transition-opacity duration-700 ${
        phase === "leaving" ? "opacity-0" : "opacity-100"
      }`}
    >
      <InspectionLogo intro className="w-36 text-neutral-200 sm:w-40" />
      <div className="mt-9 overflow-hidden">
        <h1 className="intro-word text-5xl font-light tracking-tight text-neutral-50">
          Vígil
        </h1>
      </div>
      <span className="intro-word-2 mt-4 font-mono text-[11px] uppercase tracking-[0.3em] text-neutral-500">
        b1.0.0 · beta
      </span>
    </div>
  );
}
