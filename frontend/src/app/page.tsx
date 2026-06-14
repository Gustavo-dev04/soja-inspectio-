import UploadZone from "@/components/UploadZone";

export default function HomePage() {
  return (
    <div className="max-w-xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-2">Nova Inspeção</h2>
      <p className="text-gray-500 mb-8">
        Envie uma foto dos grãos de soja para análise automática de defeitos.
      </p>
      <UploadZone />
    </div>
  );
}
