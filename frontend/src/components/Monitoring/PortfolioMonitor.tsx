import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Info as InfoIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useTheme } from '@mui/material/styles';
import { api } from '../../services/api';

interface PortfolioMonitorProps {
  portfolioId: number;
}

interface Alert {
  id: number;
  alert_type: string;
  severity: 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

interface RebalanceEvent {
  id: number;
  timestamp: string;
  status: 'PENDING' | 'COMPLETED' | 'CANCELLED';
  details: string;
}

const PortfolioMonitor: React.FC<PortfolioMonitorProps> = ({ portfolioId }) => {
  const theme = useTheme();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [rebalanceEvents, setRebalanceEvents] = useState<RebalanceEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<RebalanceEvent | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [alertsResponse, eventsResponse] = await Promise.all([
        api.get(`/portfolio/${portfolioId}/alerts`),
        api.get(`/portfolio/${portfolioId}/rebalance-events`),
      ]);
      setAlerts(alertsResponse.data);
      setRebalanceEvents(eventsResponse.data);
    } catch (err) {
      setError('Failed to fetch monitoring data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const toggleMonitoring = async () => {
    try {
      if (isMonitoring) {
        await api.post(`/portfolio/${portfolioId}/monitoring/stop`);
        setIsMonitoring(false);
      } else {
        await api.post(`/portfolio/${portfolioId}/monitoring/start`);
        setIsMonitoring(true);
      }
    } catch (err) {
      setError('Failed to toggle monitoring');
      console.error(err);
    }
  };

  const acknowledgeAlert = async (alertId: number) => {
    try {
      await api.post(`/portfolio/${portfolioId}/alerts/${alertId}/acknowledge`);
      setAlerts(alerts.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      ));
    } catch (err) {
      setError('Failed to acknowledge alert');
      console.error(err);
    }
  };

  const executeRebalance = async (eventId: number) => {
    try {
      await api.post(`/portfolio/${portfolioId}/rebalance/${eventId}/execute`);
      fetchData();
    } catch (err) {
      setError('Failed to execute rebalancing');
      console.error(err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [portfolioId]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'ERROR':
        return <ErrorIcon color="error" />;
      case 'WARNING':
        return <WarningIcon color="warning" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  return (
    <Box>
      <Box mb={3} display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="h6">Portfolio Monitor</Typography>
        <Box>
          <IconButton onClick={fetchData} size="small" sx={{ mr: 1 }}>
            <RefreshIcon />
          </IconButton>
          <Button
            variant="contained"
            color={isMonitoring ? "error" : "primary"}
            startIcon={isMonitoring ? <StopIcon /> : <PlayIcon />}
            onClick={toggleMonitoring}
          >
            {isMonitoring ? "Stop Monitoring" : "Start Monitoring"}
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Alerts
              </Typography>
              <List>
                {alerts
                  .filter(alert => !alert.acknowledged)
                  .map(alert => (
                    <ListItem
                      key={alert.id}
                      secondaryAction={
                        <Button
                          size="small"
                          onClick={() => acknowledgeAlert(alert.id)}
                        >
                          Acknowledge
                        </Button>
                      }
                    >
                      <ListItemIcon>
                        {getAlertIcon(alert.severity)}
                      </ListItemIcon>
                      <ListItemText
                        primary={alert.alert_type}
                        secondary={
                          <>
                            {alert.message}
                            <br />
                            {new Date(alert.timestamp).toLocaleString()}
                          </>
                        }
                      />
                    </ListItem>
                  ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Rebalancing Events
              </Typography>
              <List>
                {rebalanceEvents.map(event => (
                  <ListItem
                    key={event.id}
                    secondaryAction={
                      event.status === 'PENDING' && (
                        <Button
                          size="small"
                          color="primary"
                          onClick={() => executeRebalance(event.id)}
                        >
                          Execute
                        </Button>
                      )
                    }
                  >
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center">
                          <Typography variant="body1" sx={{ mr: 1 }}>
                            Rebalance Event
                          </Typography>
                          <Chip
                            label={event.status}
                            color={
                              event.status === 'COMPLETED'
                                ? 'success'
                                : event.status === 'PENDING'
                                ? 'warning'
                                : 'error'
                            }
                            size="small"
                          />
                        </Box>
                      }
                      secondary={
                        <>
                          {event.details}
                          <br />
                          {new Date(event.timestamp).toLocaleString()}
                        </>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Dialog
        open={!!selectedAlert}
        onClose={() => setSelectedAlert(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Alert Details</DialogTitle>
        <DialogContent>
          {selectedAlert && (
            <>
              <Typography variant="h6" gutterBottom>
                {selectedAlert.alert_type}
              </Typography>
              <Typography variant="body1" paragraph>
                {selectedAlert.message}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {new Date(selectedAlert.timestamp).toLocaleString()}
              </Typography>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedAlert(null)}>Close</Button>
          {selectedAlert && !selectedAlert.acknowledged && (
            <Button
              onClick={() => {
                acknowledgeAlert(selectedAlert.id);
                setSelectedAlert(null);
              }}
              color="primary"
            >
              Acknowledge
            </Button>
          )}
        </DialogActions>
      </Dialog>

      <Dialog
        open={!!selectedEvent}
        onClose={() => setSelectedEvent(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Rebalance Event Details</DialogTitle>
        <DialogContent>
          {selectedEvent && (
            <>
              <Typography variant="h6" gutterBottom>
                Status: {selectedEvent.status}
              </Typography>
              <Typography variant="body1" paragraph>
                {selectedEvent.details}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {new Date(selectedEvent.timestamp).toLocaleString()}
              </Typography>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedEvent(null)}>Close</Button>
          {selectedEvent?.status === 'PENDING' && (
            <Button
              onClick={() => {
                executeRebalance(selectedEvent.id);
                setSelectedEvent(null);
              }}
              color="primary"
            >
              Execute Rebalance
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PortfolioMonitor;
