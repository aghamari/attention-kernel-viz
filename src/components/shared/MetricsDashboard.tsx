import React from 'react';
import { PerformanceMetrics } from '../../types/attention';

interface MetricsDashboardProps {
  metrics: PerformanceMetrics;
  additionalMetrics?: { label: string; value: string | number; unit?: string }[];
}

const MetricsDashboard: React.FC<MetricsDashboardProps> = ({ metrics, additionalMetrics = [] }) => {
  const formatNumber = (num: number): string => {
    if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'G';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(2);
  };

  return (
    <div className="metrics-dashboard">
      <div className="metric-card">
        <h4>FLOPs</h4>
        <span className="metric-value">{formatNumber(metrics.flops)}</span>
      </div>

      <div className="metric-card">
        <h4>Memory Bandwidth</h4>
        <span className="metric-value">{metrics.memoryBandwidth.toFixed(2)}</span>
        <span className="metric-unit">GB</span>
      </div>

      <div className="metric-card">
        <h4>Compute Utilization</h4>
        <span className="metric-value">{(metrics.computeUtilization * 100).toFixed(1)}</span>
        <span className="metric-unit">%</span>
      </div>

      <div className="metric-card">
        <h4>Occupancy</h4>
        <span className="metric-value">{(metrics.occupancy * 100).toFixed(1)}</span>
        <span className="metric-unit">%</span>
      </div>

      <div className="metric-card">
        <h4>Arithmetic Intensity</h4>
        <span className="metric-value">{metrics.arithmeticIntensity.toFixed(1)}</span>
        <span className="metric-unit">FLOPs/byte</span>
      </div>

      {additionalMetrics.map((metric, index) => (
        <div key={index} className="metric-card">
          <h4>{metric.label}</h4>
          <span className="metric-value">{metric.value}</span>
          {metric.unit && <span className="metric-unit">{metric.unit}</span>}
        </div>
      ))}
    </div>
  );
};

export default MetricsDashboard;
