import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  IconButton,
  useTheme,
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
} from '@mui/lab';
import {
  Refresh as RefreshIcon,
  CheckCircle as HealthyIcon,
  Warning as DegradedIcon,
  Error as UnhealthyIcon,
} from '@mui/icons-material';
import { Line } from 'react-chartjs-2';

interface HealthMetric {
  type: string;
  current_value: number;
  timestamp: string;
  status: string;
  average: number;
  stddev: number;
  history_size: number;
}

interface ProviderHealth {
  provider_id: string;
  overall_status: string;
  uptime: number;
  metrics: Record<string, HealthMetric>;
}

const StatusChip: React.FC<{ status: string }> = ({ status }) => {
  const theme = useTheme();
  const getStatusColor = () => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return theme.palette.success.main;
      case 'degraded':
        return theme.palette.warning.main;
      case 'unhealthy':
        return theme.palette.error.main;
      default:
        return theme.palette.grey[500];
    }
  };

  const getStatusIcon = () => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return <HealthyIcon />;
      case 'degraded':
        return <DegradedIcon />;
      case 'unhealthy':
        return <UnhealthyIcon />;
      default:
        return null;
    }
  };

  return (
    <Chip
      icon={getStatusIcon()}
      label={status}
      sx={{
        backgroundColor: getStatusColor(),
        color: theme.palette.common.white,
      }}
    />
  );
};

const MetricCard: React.FC<{
  title: string;
  value: number;
  status: string;
  threshold?: number;
}> = ({ title, value, status, threshold = 100 }) => {
  const theme = useTheme();
  const progress = (value / threshold) * 100;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography variant="h4">
            {value.toFixed(2)}
          </Typography>
          <StatusChip status={status} />
        </Box>
        <LinearProgress
          variant="determinate"
          value={Math.min(progress, 100)}
          sx={{
            mt: 2,
            height: 8,
            borderRadius: 4,
            backgroundColor: theme.palette.grey[200],
            '& .MuiLinearProgress-bar': {
              backgroundColor: status === 'healthy'
                ? theme.palette.success.main
                : status === 'degraded'
                  ? theme.palette.warning.main
                  : theme.palette.error.main,
            },
          }}
        />
      </CardContent>
    </Card>
  );
};

const HealthTimeline: React.FC<{
  events: Array<{ timestamp: string; status: string; message: string }>;
}> = ({ events }) => {
  const theme = useTheme();

  return (
    <Timeline>
      {events.map((event, index) => (
        <TimelineItem key={index}>
          <TimelineSeparator>
            <TimelineDot
              sx={{
                backgroundColor:
                  event.status === 'healthy'
                    ? theme.palette.success.main
                    : event.status === 'degraded'
                      ? theme.palette.warning.main
                      : theme.palette.error.main,
              }}
            />
            {index < events.length - 1 && <TimelineConnector />}
          </TimelineSeparator>
          <TimelineContent>
            <Typography variant="body2" color="textSecondary">
              {new Date(event.timestamp).toLocaleTimeString()}
            </Typography>
            <Typography>{event.message}</Typography>
          </TimelineContent>
        </TimelineItem>
      ))}
    </Timeline>
  );
};

const MetricChart: React.FC<{
  data: number[];
  labels: string[];
  title: string;
}> = ({ data, labels, title }) => {
  const theme = useTheme();

  const chartData = {
    labels,
    datasets: [
      {
        label: title,
        data,
        fill: false,
        borderColor: theme.palette.primary.main,
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: title,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return <Line data={chartData} options={options} />;
};

export const HealthDashboard: React.FC = () => {
  const [healthData, setHealthData] = useState<Record<string, ProviderHealth>>({});
  const [events, setEvents] = useState<Array<any>>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);

  const connectWebSocket = useCallback(() => {
    const socket = new WebSocket('ws://localhost:8000/health/ws');

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'health_update') {
        setHealthData(data.data);
        // Add significant events to timeline
        Object.entries(data.data).forEach(([providerId, health]) => {
          if (health.overall_status !== healthData[providerId]?.overall_status) {
            setEvents(prev => [{
              timestamp: data.timestamp,
              status: health.overall_status,
              message: `Provider ${providerId} status changed to ${health.overall_status}`,
            }, ...prev].slice(0, 10));
          }
        });
      }
    };

    socket.onclose = () => {
      setTimeout(connectWebSocket, 1000);
    };

    setWs(socket);
  }, [healthData]);

  useEffect(() => {
    connectWebSocket();
    return () => ws?.close();
  }, [connectWebSocket]);

  const handleRefresh = () => {
    fetch('/api/health/providers')
      .then(response => response.json())
      .then(data => setHealthData(data))
      .catch(error => console.error('Error fetching health data:', error));
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Provider Health Dashboard</Typography>
        <IconButton onClick={handleRefresh}>
          <RefreshIcon />
        </IconButton>
      </Box>

      <Grid container spacing={3}>
        {Object.entries(healthData).map(([providerId, health]) => (
          <Grid item xs={12} key={providerId}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Typography variant="h5">{providerId}</Typography>
                  <StatusChip status={health.overall_status} />
                </Box>

                <Grid container spacing={3}>
                  {Object.entries(health.metrics).map(([metricType, metric]) => (
                    <Grid item xs={12} md={6} lg={4} key={metricType}>
                      <MetricCard
                        title={metricType}
                        value={metric.current_value}
                        status={metric.status}
                      />
                    </Grid>
                  ))}
                </Grid>

                <Box mt={3}>
                  <Typography variant="h6" gutterBottom>Metric History</Typography>
                  <Grid container spacing={3}>
                    {Object.entries(health.metrics).map(([metricType, metric]) => (
                      <Grid item xs={12} md={6} key={metricType}>
                        <MetricChart
                          title={metricType}
                          data={[metric.current_value, metric.average]}
                          labels={['Current', 'Average']}
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Event Timeline</Typography>
          <HealthTimeline events={events} />
        </CardContent>
      </Card>
    </Box>
  );
};

export default HealthDashboard;
