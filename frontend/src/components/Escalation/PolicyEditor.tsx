import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Grid,
  Typography,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Button,
  Paper,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';

interface PolicyEditorProps {
  policy?: EscalationPolicy;
  onSave: (policy: EscalationPolicy) => void;
  onCancel: () => void;
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

const ALERT_TYPES = [
  'HIGH_LATENCY',
  'ERROR_RATE',
  'LOW_HEALTH',
  'CACHE_EFFICIENCY',
  'CONNECTION_LOST',
  'API_ERROR',
];

const SEVERITY_LEVELS = [
  'WARNING',
  'ERROR',
  'CRITICAL',
];

const ESCALATION_LEVELS = [
  'L1',
  'L2',
  'L3',
  'EMERGENCY',
];

const NOTIFICATION_CHANNELS = [
  'email',
  'slack',
  'sms',
  'phone',
  'pager',
];

const AUTOMATED_ACTIONS = [
  'retry_connection',
  'failover',
  'clear_cache',
  'restart_service',
  'emergency_shutdown',
];

const PolicyEditor: React.FC<PolicyEditorProps> = ({
  policy,
  onSave,
  onCancel,
}) => {
  const [formData, setFormData] = useState<EscalationPolicy>({
    name: '',
    conditions: {
      alert_type: '',
      severity: '',
      duration_threshold: 300,
    },
    initial_level: 'L1',
    max_level: 'L3',
    escalation_delay: 900,
    notification_channels: {},
    auto_actions: {},
  });

  useEffect(() => {
    if (policy) {
      setFormData(policy);
    }
  }, [policy]);

  const handleInputChange = (field: string, value: any) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleConditionChange = (field: string, value: any) => {
    setFormData((prev) => ({
      ...prev,
      conditions: {
        ...prev.conditions,
        [field]: value,
      },
    }));
  };

  const handleChannelChange = (level: string, channels: string[]) => {
    setFormData((prev) => ({
      ...prev,
      notification_channels: {
        ...prev.notification_channels,
        [level]: channels,
      },
    }));
  };

  const handleActionChange = (level: string, actions: string[]) => {
    setFormData((prev) => ({
      ...prev,
      auto_actions: {
        ...prev.auto_actions,
        [level]: actions,
      },
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
  };

  const renderBasicInfo = () => (
    <Box mb={3}>
      <Typography variant="h6" gutterBottom>
        Basic Information
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Policy Name"
            value={formData.name}
            onChange={(e) => handleInputChange('name', e.target.value)}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>Alert Type</InputLabel>
            <Select
              value={formData.conditions.alert_type}
              onChange={(e) => handleConditionChange('alert_type', e.target.value)}
            >
              {ALERT_TYPES.map((type) => (
                <MenuItem key={type} value={type}>
                  {type}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>Severity</InputLabel>
            <Select
              value={formData.conditions.severity}
              onChange={(e) => handleConditionChange('severity', e.target.value)}
            >
              {SEVERITY_LEVELS.map((level) => (
                <MenuItem key={level} value={level}>
                  {level}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            type="number"
            label="Duration Threshold (seconds)"
            value={formData.conditions.duration_threshold}
            onChange={(e) => handleConditionChange('duration_threshold', parseInt(e.target.value))}
          />
        </Grid>
      </Grid>
    </Box>
  );

  const renderEscalationLevels = () => (
    <Box mb={3}>
      <Typography variant="h6" gutterBottom>
        Escalation Levels
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>Initial Level</InputLabel>
            <Select
              value={formData.initial_level}
              onChange={(e) => handleInputChange('initial_level', e.target.value)}
            >
              {ESCALATION_LEVELS.map((level) => (
                <MenuItem key={level} value={level}>
                  {level}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>Maximum Level</InputLabel>
            <Select
              value={formData.max_level}
              onChange={(e) => handleInputChange('max_level', e.target.value)}
            >
              {ESCALATION_LEVELS.map((level) => (
                <MenuItem key={level} value={level}>
                  {level}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            type="number"
            label="Escalation Delay (seconds)"
            value={formData.escalation_delay}
            onChange={(e) => handleInputChange('escalation_delay', parseInt(e.target.value))}
          />
        </Grid>
      </Grid>
    </Box>
  );

  const renderNotificationChannels = () => (
    <Box mb={3}>
      <Typography variant="h6" gutterBottom>
        Notification Channels
      </Typography>
      {ESCALATION_LEVELS.map((level) => (
        <Box key={level} mb={2}>
          <Typography variant="subtitle1" gutterBottom>
            {level}
          </Typography>
          <Paper variant="outlined">
            <Box p={2}>
              <Grid container spacing={1}>
                {NOTIFICATION_CHANNELS.map((channel) => {
                  const isSelected = formData.notification_channels[level]?.includes(channel);
                  return (
                    <Grid item key={channel}>
                      <Chip
                        label={channel}
                        color={isSelected ? 'primary' : 'default'}
                        onClick={() => {
                          const current = formData.notification_channels[level] || [];
                          if (isSelected) {
                            handleChannelChange(
                              level,
                              current.filter((c) => c !== channel)
                            );
                          } else {
                            handleChannelChange(level, [...current, channel]);
                          }
                        }}
                      />
                    </Grid>
                  );
                })}
              </Grid>
            </Box>
          </Paper>
        </Box>
      ))}
    </Box>
  );

  const renderAutomatedActions = () => (
    <Box mb={3}>
      <Typography variant="h6" gutterBottom>
        Automated Actions
      </Typography>
      {ESCALATION_LEVELS.map((level) => (
        <Box key={level} mb={2}>
          <Typography variant="subtitle1" gutterBottom>
            {level}
          </Typography>
          <Paper variant="outlined">
            <Box p={2}>
              <Grid container spacing={1}>
                {AUTOMATED_ACTIONS.map((action) => {
                  const isSelected = formData.auto_actions[level]?.includes(action);
                  return (
                    <Grid item key={action}>
                      <Chip
                        label={action}
                        color={isSelected ? 'primary' : 'default'}
                        onClick={() => {
                          const current = formData.auto_actions[level] || [];
                          if (isSelected) {
                            handleActionChange(
                              level,
                              current.filter((a) => a !== action)
                            );
                          } else {
                            handleActionChange(level, [...current, action]);
                          }
                        }}
                      />
                    </Grid>
                  );
                })}
              </Grid>
            </Box>
          </Paper>
        </Box>
      ))}
    </Box>
  );

  return (
    <form onSubmit={handleSubmit}>
      <Box p={2}>
        {renderBasicInfo()}
        {renderEscalationLevels()}
        {renderNotificationChannels()}
        {renderAutomatedActions()}
        
        <Box display="flex" justifyContent="flex-end" mt={3}>
          <Button
            variant="outlined"
            onClick={onCancel}
            style={{ marginRight: 8 }}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            color="primary"
            type="submit"
          >
            Save Policy
          </Button>
        </Box>
      </Box>
    </form>
  );
};

export default PolicyEditor;
