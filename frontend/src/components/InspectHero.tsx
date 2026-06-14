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
        const dataUrl = await fileToDataUrl(file);
        const result = await inspectImage(file);
        sessionStorage.setItem(
          `inspection_${result.id}`,
          JSON.stringify({ ...result, imageDataUrl: dataUrl })
        );
        router.push(`/resultado/${result.id}`);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Erro desconhecido.");
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
      className={`flex min-h-[calc(100vh-8rem)] flex-col items-center justify-center text-center transition-colors ${
        dragging ? "bg-white/[0.03]" : ""
      }`}
    >
      <button
        type="button"
        onClick={pick}
        aria-label="Inspecionar"
        className="group outline-none"
      >
        <InspectionLogo
          inspecting={loading}
          className="w-44 text-neutral-200 transition-transform duration-500 group-hover:scale-[1.03] sm:w-52"
        />
      </button>

      <h1 className="mt-10 text-3xl font-medium tracking-tight text-neutral-100 sm:text-4xl">
        {loading ? "Inspecionando…" : "Inspecionar"}
      </h1>

      <p className="mt-3 max-w-xs text-sm leading-relaxed text-neutral-500">
        {loading
          ? "Segmentando os grãos e classificando cada um."
          : "Arraste uma foto dos grãos ou clique no símbolo para começar."}
      </p>

      {!loading && (
        <button
          type="button"
          onClick={pick}
          className="mt-8 rounded-full border border-white/10 bg-white/[0.04] px-6 py-2.5 text-sm text-neutral-200 transition-colors hover:border-white/20 hover:bg-white/[0.08]"
        >
          Selecionar imagem
        </button>
      )}

      {error && <p className="mt-6 text-sm text-red-400">{error}</p>}

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
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
