import React, { useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  IconButton,
  Collapse,
  Badge
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Notifications as NotificationsIcon
} from '@mui/icons-material';
import { useWebSocket } from '../MarketDataSocket';

interface Alert {
  id: string;
  provider_id: string;
  type: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
  metric_value: number;
  threshold_value: number;
  resolved: boolean;
  resolved_at?: string;
  metadata?: Record<string, any>;
}

const severityIcons = {
  info: <InfoIcon color="info" />,
  warning: <WarningIcon color="warning" />,
  error: <ErrorIcon color="error" />,
  critical: <ErrorIcon sx={{ color: '#7B1FA2' }} />
};

const severityColors = {
  info: 'info',
  warning: 'warning',
  error: 'error',
  critical: 'secondary'
};

interface AlertItemProps {
  alert: Alert;
}

const AlertItem: React.FC<AlertItemProps> = ({ alert }) => {
  const [expanded, setExpanded] = useState(false);
  const formattedTime = new Date(alert.timestamp).toLocaleString();

  return (
    <ListItem
      sx={{
        border: 1,
        borderColor: 'divider',
        borderRadius: 1,
        mb: 1,
        flexDirection: 'column',
        alignItems: 'stretch'
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
        <ListItemIcon>{severityIcons[alert.severity]}</ListItemIcon>
        <ListItemText
          primary={alert.message}
          secondary={`${alert.provider_id} - ${formattedTime}`}
        />
        <Chip
          label={alert.type.replace('_', ' ')}
          color={severityColors[alert.severity] as any}
          size="small"
          sx={{ mr: 1 }}
        />
        <IconButton size="small" onClick={() => setExpanded(!expanded)}>
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>
      
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <Box sx={{ p: 2, bgcolor: 'background.default' }}>
          <Typography variant="body2" gutterBottom>
            Metric Value: {alert.metric_value.toFixed(2)}
          </Typography>
          <Typography variant="body2" gutterBottom>
            Threshold: {alert.threshold_value.toFixed(2)}
          </Typography>
          {alert.resolved && (
            <Typography variant="body2" color="success.main">
              Resolved at: {new Date(alert.resolved_at!).toLocaleString()}
            </Typography>
          )}
          {alert.metadata && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="body2" color="text.secondary">
                Additional Information:
              </Typography>
              <pre style={{ margin: 0, fontSize: '0.875rem' }}>
                {JSON.stringify(alert.metadata, null, 2)}
              </pre>
            </Box>
          )}
        </Box>
      </Collapse>
    </ListItem>
  );
};

const AlertPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [activeCount, setActiveCount] = useState(0);
  const socket = useWebSocket();

  useEffect(() => {
    if (!socket) return;

    const handleAlert = (data: Alert) => {
      setAlerts(prev => {
        const newAlerts = [data, ...prev].slice(0, 100); // Keep last 100 alerts
        setActiveCount(newAlerts.filter(a => !a.resolved).length);
        return newAlerts;
      });
    };

    socket.on('alert', handleAlert);

    // Fetch initial alerts
    fetch('/api/performance/alerts')
      .then(res => res.json())
      .then(data => {
        setAlerts(data);
        setActiveCount(data.filter((a: Alert) => !a.resolved).length);
      })
      .catch(console.error);

    return () => {
      socket.off('alert', handleAlert);
    };
  }, [socket]);

  return (
    <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Badge badgeContent={activeCount} color="error" sx={{ mr: 1 }}>
          <NotificationsIcon />
        </Badge>
        <Typography variant="h6">
          Performance Alerts
        </Typography>
      </Box>
      
      <List sx={{ maxHeight: 400, overflow: 'auto' }}>
        {alerts.map(alert => (
          <AlertItem key={alert.id} alert={alert} />
        ))}
      </List>
    </Paper>
  );
};

export default AlertPanel;
