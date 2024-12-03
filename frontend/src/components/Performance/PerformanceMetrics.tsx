import React from 'react';
import { Box, Paper, Typography } from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

interface MetricChartProps {
  title: string;
  data: Array<{
    timestamp: string;
    value: number;
  }>;
  yAxisLabel: string;
  color?: string;
}

const MetricChart: React.FC<MetricChartProps> = ({
  title,
  data,
  yAxisLabel,
  color = '#8884d8'
}) => (
  <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
    <Typography variant="h6" gutterBottom>
      {title}
    </Typography>
    <Box sx={{ height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(value) => new Date(value).toLocaleTimeString()}
          />
          <YAxis label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }} />
          <Tooltip
            labelFormatter={(value) => new Date(value).toLocaleString()}
            formatter={(value: number) => [value.toFixed(2), title]}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            name={title}
            dot={false}
            activeDot={{ r: 8 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  </Paper>
);

interface PerformanceMetricsProps {
  providerId: string;
  metrics: {
    latency: Array<{ timestamp: string; value: number }>;
    errorRate: Array<{ timestamp: string; value: number }>;
    cacheHitRate: Array<{ timestamp: string; value: number }>;
    healthScore: Array<{ timestamp: string; value: number }>;
  };
}

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ providerId, metrics }) => {
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        {providerId} - Detailed Metrics
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
        <MetricChart
          title="Latency"
          data={metrics.latency}
          yAxisLabel="ms"
          color="#8884d8"
        />
        <MetricChart
          title="Error Rate"
          data={metrics.errorRate}
          yAxisLabel="errors/min"
          color="#ff4444"
        />
        <MetricChart
          title="Cache Hit Rate"
          data={metrics.cacheHitRate}
          yAxisLabel="%"
          color="#4CAF50"
        />
        <MetricChart
          title="Health Score"
          data={metrics.healthScore}
          yAxisLabel="%"
          color="#2196F3"
        />
      </Box>
    </Box>
  );
};

export default PerformanceMetrics;
