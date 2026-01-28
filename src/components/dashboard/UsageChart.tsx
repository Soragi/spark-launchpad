import { forwardRef, useMemo } from "react";

interface UsageChartProps {
  data: number[];
  max: number;
  color: string;
  gradientId?: string;
}

const UsageChart = forwardRef<SVGSVGElement, UsageChartProps>(
  ({ data, max, color, gradientId = "chartGradient" }, ref) => {
    const width = 200;
    const height = 80;
    const padding = 8;

    // Ensure we have at least 2 points for the chart
    const chartData = useMemo(() => {
      if (data.length === 0) return [0, 0];
      if (data.length === 1) return [data[0], data[0]];
      return data;
    }, [data]);

    const points = useMemo(() => {
      return chartData.map((value, index) => {
        const x = padding + (index / (chartData.length - 1)) * (width - 2 * padding);
        const y = height - padding - (Math.min(value, max) / max) * (height - 2 * padding);
        return { x, y, value };
      });
    }, [chartData, max]);

    const linePath = useMemo(() => {
      if (points.length < 2) return "";
      
      // Create smooth curve using cubic bezier
      let path = `M ${points[0].x} ${points[0].y}`;
      
      for (let i = 1; i < points.length; i++) {
        const prev = points[i - 1];
        const curr = points[i];
        const cpx = (prev.x + curr.x) / 2;
        path += ` C ${cpx} ${prev.y}, ${cpx} ${curr.y}, ${curr.x} ${curr.y}`;
      }
      
      return path;
    }, [points]);

    const areaPath = useMemo(() => {
      if (points.length < 2) return "";
      return `${linePath} L ${points[points.length - 1].x} ${height - padding} L ${points[0].x} ${height - padding} Z`;
    }, [linePath, points]);

    const latestValue = chartData[chartData.length - 1] || 0;
    const latestPoint = points[points.length - 1];

    return (
      <svg 
        ref={ref} 
        viewBox={`0 0 ${width} ${height}`} 
        className="w-full h-full"
        style={{ overflow: "visible" }}
      >
        <defs>
          {/* Gradient for area fill */}
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0.02" />
          </linearGradient>
          
          {/* Glow filter */}
          <filter id={`${gradientId}-glow`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Subtle grid lines */}
        {[0, 50, 100].map((pct) => {
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
              strokeOpacity="0.3"
            />
          );
        })}

        {/* Area fill with gradient */}
        <path
          d={areaPath}
          fill={`url(#${gradientId})`}
          className="transition-all duration-500 ease-out"
        />

        {/* Main line with glow */}
        <path
          d={linePath}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter={`url(#${gradientId}-glow)`}
          className="transition-all duration-500 ease-out"
        />

        {/* Animated pulse on latest point */}
        {latestPoint && (
          <>
            {/* Pulse ring */}
            <circle
              cx={latestPoint.x}
              cy={latestPoint.y}
              r="6"
              fill="none"
              stroke={color}
              strokeWidth="1"
              opacity="0.4"
              className="animate-ping"
              style={{ transformOrigin: `${latestPoint.x}px ${latestPoint.y}px` }}
            />
            {/* Solid point */}
            <circle
              cx={latestPoint.x}
              cy={latestPoint.y}
              r="4"
              fill={color}
              className="transition-all duration-300 ease-out"
              style={{ filter: `drop-shadow(0 0 4px ${color})` }}
            />
            <circle
              cx={latestPoint.x}
              cy={latestPoint.y}
              r="2"
              fill="hsl(var(--background))"
            />
          </>
        )}
      </svg>
    );
  }
);

UsageChart.displayName = "UsageChart";

export default UsageChart;
