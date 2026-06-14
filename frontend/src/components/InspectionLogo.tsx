type Props = {
  /** Liga a varredura: palitinhos rápidos + íris/arco lentos (horário). */
  inspecting?: boolean;
  /** Liga a animação de entrada (desenha os anéis + revela os palitinhos). */
  intro?: boolean;
  className?: string;
};

// 12 palitinhos a cada 30° + um anel fino de 24 micro-ticks (instrumento).
const SPOKES = Array.from({ length: 12 }, (_, i) => i * 30);
const TICKS = Array.from({ length: 24 }, (_, i) => i * 15);

/**
 * Símbolo da Vígil: uma íris/abertura de precisão — pupila, anéis, um bezel de
 * micro-ticks e palitinhos radiais, com um arco de varredura. Parado é sóbrio;
 * em `inspecting` os palitinhos giram rápido e a íris devagar (sentido horário).
 */
export default function InspectionLogo({
  inspecting = false,
  intro = false,
  className = "",
}: Props) {
  return (
    <div className={`relative aspect-square ${intro ? "logo-intro" : ""} ${className}`}>
      {/* camada rápida — palitinhos + micro-ticks */}
      <svg
        viewBox="0 0 200 200"
        fill="none"
        className={`absolute inset-0 h-full w-full ${inspecting ? "spin-fast" : ""}`}
      >
        <g className="text-neutral-200">
          {SPOKES.map((d) => (
            <line
              key={`s${d}`}
              x1="100"
              y1="14"
              x2="100"
              y2="40"
              transform={`rotate(${d} 100 100)`}
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              className={intro ? "intro-fade" : ""}
              style={intro ? { animationDelay: `${0.45 + (d / 30) * 0.035}s` } : undefined}
            />
          ))}
        </g>
        <g className="text-neutral-500" opacity="0.55">
          {TICKS.map((d) => (
            <line
              key={`t${d}`}
              x1="100"
              y1="47"
              x2="100"
              y2="51"
              transform={`rotate(${d} 100 100)`}
              stroke="currentColor"
              strokeWidth="1"
              strokeLinecap="round"
            />
          ))}
        </g>
      </svg>

      {/* camada lenta — anéis, arco de varredura e pupila */}
      <svg
        viewBox="0 0 200 200"
        fill="none"
        className={`absolute inset-0 h-full w-full ${inspecting ? "spin-slow" : ""}`}
      >
        <circle
          cx="100"
          cy="100"
          r="46"
          pathLength={1}
          stroke="currentColor"
          strokeWidth="1.25"
          className={`text-neutral-700 ${intro ? "intro-draw" : ""}`}
        />
        <circle
          cx="100"
          cy="100"
          r="40"
          pathLength={1}
          stroke="currentColor"
          strokeWidth="2.5"
          className={`text-neutral-200 ${intro ? "intro-draw" : ""}`}
          style={intro ? { animationDelay: "0.15s" } : undefined}
        />
        <circle
          cx="100"
          cy="100"
          r="12"
          pathLength={1}
          stroke="currentColor"
          strokeWidth="2.5"
          className={`text-neutral-200 ${intro ? "intro-draw" : ""}`}
          style={intro ? { animationDelay: "0.35s" } : undefined}
        />
        {/* arco de varredura — acende em verde ao inspecionar */}
        <path
          d="M83.1 63.75 A40 40 0 0 1 116.9 63.75"
          pathLength={1}
          stroke="currentColor"
          strokeWidth="3.4"
          strokeLinecap="round"
          className={`transition-colors duration-500 ${
            inspecting ? "text-brand" : "text-neutral-600"
          } ${intro ? "intro-draw" : ""}`}
          style={intro ? { animationDelay: "0.55s" } : undefined}
        />
        <circle cx="100" cy="100" r="1.8" fill="currentColor" className="text-neutral-200" />
      </svg>
    </div>
  );
}
