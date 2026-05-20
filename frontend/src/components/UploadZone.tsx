"use client";
import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { inspectImage } from "@/lib/api";

export default function UploadZone() {
  const router = useRouter();
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) {
        setError("Por favor, envie uma imagem (JPEG, PNG).");
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
      } finally {
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

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`border-2 border-dashed rounded-2xl p-12 text-center transition-colors ${
        dragging
          ? "border-brand bg-green-50"
          : "border-gray-300 hover:border-brand"
      }`}
    >
      {loading ? (
        <p className="text-brand font-semibold animate-pulse">
          Analisando grãos...
        </p>
      ) : (
        <>
          <p className="text-gray-600 mb-4">
            Arraste uma imagem ou clique para selecionar
          </p>
          <label className="cursor-pointer bg-brand text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors">
            Selecionar Imagem
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) =>
                e.target.files?.[0] && handleFile(e.target.files[0])
              }
            />
          </label>
        </>
      )}
      {error && <p className="mt-4 text-red-500 text-sm">{error}</p>}
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
