import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  List,
  ListItem,
  ListItemText,
  Chip,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
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
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';

interface EscalationEvent {
  id: string;
  alert: {
    provider_id: string;
    type: string;
    severity: string;
    message: string;
  };
  policy: {
    name: string;
    initial_level: string;
    max_level: string;
  };
  current_level: string;
  start_time: string;
  last_escalation: string;
  actions_taken: string[];
  resolved: boolean;
  resolution_time?: string;
}

interface EscalationPolicy {
  name: string;
  conditions: {
    alert_type: string;
    severity: string;
    duration_threshold: number;
  };
  initial_level: string;
  max_level: string;
  escalation_delay: number;
  notification_channels: Record<string, string[]>;
  auto_actions: Record<string, string[]>;
}

const EscalationDashboard: React.FC = () => {
  const theme = useTheme();
  const [activeEscalations, setActiveEscalations] = useState<EscalationEvent[]>([]);
  const [policies, setPolicies] = useState<EscalationPolicy[]>([]);
  const [selectedPolicy, setSelectedPolicy] = useState<EscalationPolicy | null>(null);
  const [isPolicyDialogOpen, setIsPolicyDialogOpen] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [escalationsRes, policiesRes] = await Promise.all([
          fetch('/api/escalation/active'),
          fetch('/api/escalation/policies'),
        ]);
        
        const escalationsData = await escalationsRes.json();
        const policiesData = await policiesRes.json();
        
        setActiveEscalations(escalationsData);
        setPolicies(policiesData);
      } catch (error) {
        console.error('Error fetching escalation data:', error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const renderSeverityIcon = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'resolved':
        return <CheckCircleIcon color="success" />;
      default:
        return null;
    }
  };

  const renderActiveEscalations = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Active Escalations
        </Typography>
        <Timeline>
          {activeEscalations.map((event) => (
            <TimelineItem key={event.id}>
              <TimelineSeparator>
                <TimelineDot color={event.resolved ? 'success' : 'error'} />
                <TimelineConnector />
              </TimelineSeparator>
              <TimelineContent>
                <Box mb={2}>
                  <Typography variant="subtitle1" display="flex" alignItems="center">
                    {renderSeverityIcon(event.alert.severity)}
                    <Box ml={1}>{event.policy.name}</Box>
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {event.alert.provider_id} - {event.alert.type}
                  </Typography>
                  <Box mt={1}>
                    <Chip
                      label={`Level ${event.current_level}`}
                      color={event.resolved ? 'success' : 'error'}
                      size="small"
                      style={{ marginRight: 4 }}
                    />
                    <Chip
                      label={event.resolved ? 'Resolved' : 'Active'}
                      variant="outlined"
                      size="small"
                    />
                  </Box>
                  <Box mt={1}>
                    <Typography variant="body2">
                      Started: {new Date(event.start_time).toLocaleString()}
                    </Typography>
                    {event.resolved && (
                      <Typography variant="body2">
                        Resolved: {new Date(event.resolution_time!).toLocaleString()}
                      </Typography>
                    )}
                  </Box>
                  {event.actions_taken.length > 0 && (
                    <Box mt={1}>
                      <Typography variant="body2">Actions Taken:</Typography>
                      <List dense>
                        {event.actions_taken.map((action, index) => (
                          <ListItem key={index}>
                            <ListItemText primary={action} />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}
                </Box>
              </TimelineContent>
            </TimelineItem>
          ))}
        </Timeline>
      </CardContent>
    </Card>
  );

  const renderPolicies = () => (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Escalation Policies
          </Typography>
          <Button
            variant="contained"
            color="primary"
            onClick={() => {
              setSelectedPolicy(null);
              setIsPolicyDialogOpen(true);
            }}
          >
            Add Policy
          </Button>
        </Box>
        <List>
          {policies.map((policy) => (
            <ListItem
              key={policy.name}
              secondaryAction={
                <Box>
                  <IconButton
                    edge="end"
                    aria-label="edit"
                    onClick={() => {
                      setSelectedPolicy(policy);
                      setIsPolicyDialogOpen(true);
                    }}
                  >
                    <EditIcon />
                  </IconButton>
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => {/* Implement delete */}}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Box>
              }
            >
              <ListItemText
                primary={policy.name}
                secondary={
                  <>
                    <Typography variant="body2">
                      {policy.conditions.alert_type} - {policy.conditions.severity}
                    </Typography>
                    <Typography variant="body2">
                      Levels: {policy.initial_level} → {policy.max_level}
                    </Typography>
                  </>
                }
              />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );

  const renderMetrics = () => {
    const levelData = activeEscalations.reduce((acc: Record<string, number>, event) => {
      acc[event.current_level] = (acc[event.current_level] || 0) + 1;
      return acc;
    }, {});

    const pieData = Object.entries(levelData).map(([name, value]) => ({
      name,
      value,
    }));

    const COLORS = [
      theme.palette.primary.main,
      theme.palette.secondary.main,
      theme.palette.error.main,
      theme.palette.warning.main,
    ];

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Escalation Metrics
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box height={300}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label
                    >
                      {pieData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={COLORS[index % COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box height={300}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={activeEscalations.map((event) => ({
                      time: new Date(event.start_time).getTime(),
                      level: event.current_level,
                    }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="time"
                      type="number"
                      domain={['auto', 'auto']}
                      tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip
                      labelFormatter={(time) => new Date(time).toLocaleString()}
                    />
                    <Line
                      type="monotone"
                      dataKey="level"
                      stroke={theme.palette.primary.main}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  };

  return (
    <Box p={3}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          {renderActiveEscalations()}
        </Grid>
        <Grid item xs={12} md={4}>
          {renderPolicies()}
        </Grid>
        <Grid item xs={12}>
          {renderMetrics()}
        </Grid>
      </Grid>

      <Dialog
        open={isPolicyDialogOpen}
        onClose={() => setIsPolicyDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {selectedPolicy ? 'Edit Policy' : 'New Policy'}
        </DialogTitle>
        <DialogContent>
          {/* Add policy editor form */}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsPolicyDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="contained"
            color="primary"
            onClick={() => {
              // Implement save
              setIsPolicyDialogOpen(false);
            }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default EscalationDashboard;
