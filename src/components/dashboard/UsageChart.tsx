interface UsageChartProps {
  data: number[];
  max: number;
  color: string;
}

const UsageChart = ({ data, max, color }: UsageChartProps) => {
  const width = 200;
  const height = 100;
  const padding = 10;

  const points = data.map((value, index) => {
    const x = padding + (index / (data.length - 1)) * (width - 2 * padding);
    const y = height - padding - (value / max) * (height - 2 * padding);
    return `${x},${y}`;
  });

  const areaPoints = [
    `${padding},${height - padding}`,
    ...points,
    `${width - padding},${height - padding}`,
  ].join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
      {/* Grid lines */}
      {[0, 25, 50, 75, 100].map((pct) => {
        const y = height - padding - (pct / 100) * (height - 2 * padding);
        return (
          <line
            key={pct}
            x1={padding}
            y1={y}
            x2={width - padding}
            y2={y}
            stroke="hsl(var(--border))"
            strokeWidth="0.5"
            strokeDasharray="2,2"
          />
        );
      })}

      {/* Area fill */}
      <polygon
        points={areaPoints}
        fill={color}
        fillOpacity="0.1"
      />

      {/* Line */}
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Data points */}
      {data.map((value, index) => {
        const x = padding + (index / (data.length - 1)) * (width - 2 * padding);
        const y = height - padding - (value / max) * (height - 2 * padding);
        return (
          <circle
            key={index}
            cx={x}
            cy={y}
            r="3"
            fill="hsl(var(--background))"
            stroke={color}
            strokeWidth="2"
          />
        );
      })}
    </svg>
  );
};

export default UsageChart;
