type Props = {
  /** Liga a animação de varredura (palitinhos rápidos, íris lenta). */
  inspecting?: boolean;
  className?: string;
};

// 12 palitinhos a cada 30°, irradiando do centro.
const SPOKES = Array.from({ length: 12 }, (_, i) => i * 30);

/**
 * Símbolo de marca: uma íris central (anel + pupila) cercada por palitinhos
 * radiais. Parado é estático e sóbrio; em `inspecting` os palitinhos giram
 * rápido em sentido horário e a íris gira devagar, dando sensação de varredura.
 */
export default function InspectionLogo({
  inspecting = false,
  className = "",
}: Props) {
  return (
    <div className={`relative aspect-square ${className}`}>
      {/* palitinhos — camada rápida */}
      <svg
        viewBox="0 0 200 200"
        fill="none"
        className={`absolute inset-0 h-full w-full ${
          inspecting ? "spin-fast" : ""
        }`}
      >
        {SPOKES.map((deg) => (
          <line
            key={deg}
            x1="100"
            y1="16"
            x2="100"
            y2="42"
            transform={`rotate(${deg} 100 100)`}
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            className="text-neutral-200"
          />
        ))}
      </svg>

      {/* íris — camada lenta */}
      <svg
        viewBox="0 0 200 200"
        fill="none"
        className={`absolute inset-0 h-full w-full ${
          inspecting ? "spin-slow" : ""
        }`}
      >
        <circle
          cx="100"
          cy="100"
          r="40"
          stroke="currentColor"
          strokeWidth="3"
          className="text-neutral-200"
        />
        <circle
          cx="100"
          cy="100"
          r="15"
          stroke="currentColor"
          strokeWidth="3"
          className="text-neutral-200"
        />
        {/* arco de varredura — pista do giro lento; acende ao inspecionar */}
        <path
          d="M83.1 63.75 A40 40 0 0 1 116.9 63.75"
          stroke="currentColor"
          strokeWidth="3.4"
          strokeLinecap="round"
          className={`transition-colors duration-500 ${
            inspecting ? "text-brand" : "text-neutral-700"
          }`}
        />
      </svg>
    </div>
  );
}
