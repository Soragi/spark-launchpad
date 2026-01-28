import { forwardRef, useMemo } from "react";

interface GaugeChartProps {
  value: number;
  max: number;
  label: string;
  unit?: string;
  size?: number;
  showPercentage?: boolean;
}

const GaugeChart = forwardRef<HTMLDivElement, GaugeChartProps>(
  ({ value, max, label, unit = "", size = 160, showPercentage = false }, ref) => {
    const percentage = useMemo(() => Math.min((value / max) * 100, 100), [value, max]);
    const strokeWidth = 10;
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * Math.PI;
    const offset = circumference - (percentage / 100) * circumference;

    // Color based on usage percentage with smooth gradient
    const getColor = (pct: number) => {
      if (pct < 50) return "hsl(var(--gauge-green))";
      if (pct < 80) return "hsl(var(--gauge-yellow))";
      return "hsl(var(--gauge-red))";
    };

    const color = getColor(percentage);
    const displayValue = showPercentage 
      ? `${Math.round(percentage)}%` 
      : `${value.toFixed(1)} ${unit}`;

    const gradientId = `gauge-gradient-${label.replace(/\s/g, '-')}`;

    return (
      <div ref={ref} className="flex flex-col items-center">
        <h3 className="text-xs font-medium text-muted-foreground mb-3 uppercase tracking-wider">
          {label}
        </h3>
        <div className="relative" style={{ width: size, height: size / 2 + 24 }}>
          <svg
            width={size}
            height={size / 2 + strokeWidth}
            className="overflow-visible"
          >
            <defs>
              {/* Gradient for the arc */}
              <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor={color} stopOpacity="0.6" />
                <stop offset="100%" stopColor={color} stopOpacity="1" />
              </linearGradient>
              
              {/* Glow filter */}
              <filter id={`${gradientId}-glow`} x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Background arc with subtle pattern */}
            <path
              d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
              fill="none"
              stroke="hsl(var(--gauge-bg))"
              strokeWidth={strokeWidth}
              strokeLinecap="round"
            />
            
            {/* Tick marks */}
            {[0, 25, 50, 75, 100].map((tick) => {
              const angle = Math.PI - (tick / 100) * Math.PI;
              const innerR = radius - strokeWidth / 2 - 4;
              const outerR = radius - strokeWidth / 2 - 8;
              const x1 = size / 2 + innerR * Math.cos(angle);
              const y1 = size / 2 - innerR * Math.sin(angle);
              const x2 = size / 2 + outerR * Math.cos(angle);
              const y2 = size / 2 - outerR * Math.sin(angle);
              return (
                <line
                  key={tick}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth="1"
                  strokeOpacity="0.3"
                />
              );
            })}

            {/* Value arc with glow */}
            <path
              d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
              fill="none"
              stroke={`url(#${gradientId})`}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              filter={`url(#${gradientId}-glow)`}
              className="transition-all duration-500 ease-out"
            />

            {/* Animated end cap */}
            {percentage > 0 && (
              <>
                {(() => {
                  const angle = Math.PI - (percentage / 100) * Math.PI;
                  const cx = size / 2 + radius * Math.cos(angle);
                  const cy = size / 2 - radius * Math.sin(angle);
                  return (
                    <>
                      <circle
                        cx={cx}
                        cy={cy}
                        r="6"
                        fill="none"
                        stroke={color}
                        strokeWidth="1"
                        opacity="0.4"
                        className="animate-ping"
                        style={{ transformOrigin: `${cx}px ${cy}px` }}
                      />
                      <circle
                        cx={cx}
                        cy={cy}
                        r="3"
                        fill={color}
                        style={{ filter: `drop-shadow(0 0 6px ${color})` }}
                        className="transition-all duration-300"
                      />
                    </>
                  );
                })()}
              </>
            )}
          </svg>
          
          {/* Center value display */}
          <div className="absolute inset-0 flex flex-col items-center justify-end pb-1">
            <span
              className="text-2xl font-bold tabular-nums transition-colors duration-300"
              style={{ color }}
            >
              {displayValue}
            </span>
            {!showPercentage && (
              <span className="text-[10px] text-muted-foreground mt-0.5">
                of {max} {unit}
              </span>
            )}
          </div>
        </div>
      </div>
    );
  }
);

GaugeChart.displayName = "GaugeChart";

export default GaugeChart;
