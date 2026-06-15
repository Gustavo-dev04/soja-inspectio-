"use client";
import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { inspectImage } from "@/lib/api";
import InspectionLogo from "@/components/InspectionLogo";

export default function InspectHero() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) {
        setError("Envie uma imagem (JPEG, PNG).");
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const dataUrl = await fileToDataUrl(file); // leitura única, reusada abaixo
        const result = await inspectImage(dataUrl);
        sessionStorage.setItem(
          `inspection_${result.id}`,
          JSON.stringify({ ...result, imageDataUrl: dataUrl })
        );
        router.push(`/resultado/${result.id}`);
      } catch (e: unknown) {
        console.error("Falha na inspeção:", e);
        const msg =
          e instanceof Error
            ? e.message
            : typeof e === "string"
            ? e
            : "Erro inesperado ao inspecionar.";
        setError(msg);
        setLoading(false);
      }
    },
    [router]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const pick = () => !loading && inputRef.current?.click();

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`relative flex min-h-[calc(100vh-9rem)] flex-col items-center justify-center text-center transition-colors ${
        dragging ? "bg-white/[0.03]" : ""
      }`}
    >
      <button
        type="button"
        onClick={pick}
        aria-label="Inspecionar"
        className="group outline-none focus-visible:ring-2 focus-visible:ring-brand/40 rounded-full"
      >
        <InspectionLogo
          inspecting={loading}
          className="w-40 text-neutral-200 transition-transform duration-500 group-hover:scale-[1.03] sm:w-48"
        />
      </button>

      <div className="mt-9 flex items-center gap-3">
        <h1 className="text-4xl font-light tracking-tight text-neutral-50 sm:text-5xl">
          Vígil
        </h1>
        <span className="rounded-full border border-brand/30 bg-brand/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.2em] text-brand">
          beta
        </span>
      </div>

      <p className="mt-3 text-sm text-neutral-500">
        {loading
          ? "Segmentando os grãos e classificando cada um."
          : "Inspeção visual de grãos de soja."}
      </p>

      {loading ? (
        <p className="mt-9 animate-pulse text-sm text-neutral-300">Inspecionando…</p>
      ) : (
        <>
          <button
            type="button"
            onClick={pick}
            className="mt-9 rounded-full bg-neutral-100 px-7 py-2.5 text-sm font-medium text-neutral-900 transition-colors hover:bg-white"
          >
            Inspecionar
          </button>
          <p className="mt-4 text-xs text-neutral-600">ou arraste uma foto aqui</p>
        </>
      )}

      {error && <p className="mt-6 text-sm text-red-400">{error}</p>}

      <span className="pointer-events-none absolute bottom-2 font-mono text-[10px] tracking-wider text-neutral-700">
        b1.0.0
      </span>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
    </div>
  );
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Não consegui ler a imagem selecionada."));
    reader.readAsDataURL(file);
  });
}
