interface GaugeChartProps {
  value: number;
  max: number;
  label: string;
  unit?: string;
  size?: number;
  showPercentage?: boolean;
}

const GaugeChart = ({
  value,
  max,
  label,
  unit = "",
  size = 180,
  showPercentage = false,
}: GaugeChartProps) => {
  const percentage = Math.min((value / max) * 100, 100);
  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;

  // Color based on usage percentage
  const getColor = (pct: number) => {
    if (pct < 50) return "hsl(var(--gauge-green))";
    if (pct < 80) return "hsl(var(--gauge-yellow))";
    return "hsl(var(--gauge-red))";
  };

  const displayValue = showPercentage ? `${Math.round(percentage)} %` : `${value.toFixed(2)} ${unit}`;

  return (
    <div className="flex flex-col items-center">
      <h3 className="text-sm font-medium text-muted-foreground mb-4">{label}</h3>
      <div className="relative" style={{ width: size, height: size / 2 + 20 }}>
        <svg
          width={size}
          height={size / 2 + strokeWidth}
          className="transform -rotate-0"
        >
          {/* Background arc */}
          <path
            d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
            fill="none"
            stroke="hsl(var(--gauge-bg))"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {/* Value arc */}
          <path
            d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
            fill="none"
            stroke={getColor(percentage)}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="gauge-arc"
            style={{
              filter: `drop-shadow(0 0 6px ${getColor(percentage)})`,
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-end pb-2">
          <span
            className="text-2xl font-bold"
            style={{ color: getColor(percentage) }}
          >
            {displayValue}
          </span>
          {!showPercentage && (
            <span className="text-xs text-muted-foreground">
              {max} {unit} available
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default GaugeChart;
