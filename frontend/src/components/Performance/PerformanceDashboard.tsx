import React, { useEffect, useState } from 'react';
import { Box, Grid, Paper, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { useWebSocket } from '../MarketDataSocket';
import AlertPanel from './AlertPanel';

interface ProviderMetrics {
  health: {
    status: string;
    score: number;
    lastUpdate: string;
  };
  latency: {
    average: number;
    p95: number;
    p99: number;
    trend: number[];
  };
  cacheEfficiency: {
    hitRate: number;
    missRate: number;
    evictionRate: number;
    size: number;
  };
  errorRate: {
    total: number;
    byType: Record<string, number>;
    trend: number[];
  };
}

interface PerformanceData {
  timestamp: string;
  providerId: string;
  metrics: ProviderMetrics;
}

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  trend?: number[];
}> = ({ title, value, trend }) => (
  <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
    <Typography variant="h6" gutterBottom>
      {title}
    </Typography>
    <Typography variant="h4">{value}</Typography>
    {trend && trend.length > 0 && (
      <Box sx={{ mt: 2, height: 100 }}>
        <LineChart width={200} height={100} data={trend.map((v, i) => ({ value: v, index: i }))}>
          <Line type="monotone" dataKey="value" stroke="#8884d8" dot={false} />
        </LineChart>
      </Box>
    )}
  </Paper>
);

const PerformanceDashboard: React.FC = () => {
  const [performanceData, setPerformanceData] = useState<Record<string, PerformanceData>>({});
  const socket = useWebSocket();

  useEffect(() => {
    if (!socket) return;

    const handlePerformanceUpdate = (data: PerformanceData) => {
      setPerformanceData(prev => ({
        ...prev,
        [data.providerId]: data
      }));
    };

    socket.on('performance_update', handlePerformanceUpdate);

    // Initial data fetch
    fetch('/api/performance/current')
      .then(res => res.json())
      .then(data => setPerformanceData(data))
      .catch(console.error);

    return () => {
      socket.off('performance_update', handlePerformanceUpdate);
    };
  }, [socket]);

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Provider Performance Dashboard
      </Typography>
      
      {/* Alert Panel */}
      <Box sx={{ mb: 4 }}>
        <AlertPanel />
      </Box>
      
      {Object.entries(performanceData).map(([providerId, data]) => (
        <Box key={providerId} sx={{ mb: 4 }}>
          <Typography variant="h5" gutterBottom>
            {providerId}
          </Typography>
          <Grid container spacing={3}>
            {/* Health Status */}
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Health Score"
                value={`${data.metrics.health.score}%`}
              />
            </Grid>

            {/* Latency */}
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Avg Latency"
                value={`${data.metrics.latency.average}ms`}
                trend={data.metrics.latency.trend}
              />
            </Grid>

            {/* Cache Efficiency */}
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Cache Hit Rate"
                value={`${(data.metrics.cacheEfficiency.hitRate * 100).toFixed(1)}%`}
              />
            </Grid>

            {/* Error Rate */}
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Error Rate"
                value={`${data.metrics.errorRate.total}`}
                trend={data.metrics.errorRate.trend}
              />
            </Grid>
          </Grid>
        </Box>
      ))}
    </Box>
  );
};

export default PerformanceDashboard;
