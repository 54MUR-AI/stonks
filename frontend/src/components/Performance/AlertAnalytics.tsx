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
  LinearProgress,
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
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface AlertPattern {
  id: string;
  alert_types: string[];
  provider_ids: string[];
  root_cause_probability: { [key: string]: number };
  first_occurrence: string;
  last_occurrence: string;
  alerts: any[];
}

interface AnomalyPrediction {
  provider_id: string;
  alert_type: string;
  probability: number;
  predicted_value: number;
  prediction_time: string;
  features: { [key: string]: number };
}

const AlertAnalytics: React.FC = () => {
  const theme = useTheme();
  const [patterns, setPatterns] = useState<AlertPattern[]>([]);
  const [predictions, setPredictions] = useState<AnomalyPrediction[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [patternsRes, predictionsRes] = await Promise.all([
          fetch('/api/alerts/patterns'),
          fetch('/api/alerts/predictions'),
        ]);
        
        const patternsData = await patternsRes.json();
        const predictionsData = await predictionsRes.json();
        
        setPatterns(patternsData);
        setPredictions(predictionsData);
      } catch (error) {
        console.error('Error fetching analytics data:', error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []);

  const renderPatternTimeline = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Alert Patterns
        </Typography>
        <Timeline>
          {patterns.map((pattern) => (
            <TimelineItem key={pattern.id}>
              <TimelineSeparator>
                <TimelineDot color="primary" />
                <TimelineConnector />
              </TimelineSeparator>
              <TimelineContent>
                <Box mb={2}>
                  <Typography variant="subtitle1">
                    Pattern {pattern.id}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {new Date(pattern.first_occurrence).toLocaleString()} -{' '}
                    {new Date(pattern.last_occurrence).toLocaleString()}
                  </Typography>
                  <Box mt={1}>
                    {pattern.alert_types.map((type) => (
                      <Chip
                        key={type}
                        label={type}
                        size="small"
                        style={{ marginRight: 4, marginBottom: 4 }}
                      />
                    ))}
                  </Box>
                  <Box mt={1}>
                    {pattern.provider_ids.map((id) => (
                      <Chip
                        key={id}
                        label={id}
                        variant="outlined"
                        size="small"
                        style={{ marginRight: 4, marginBottom: 4 }}
                      />
                    ))}
                  </Box>
                </Box>
              </TimelineContent>
            </TimelineItem>
          ))}
        </Timeline>
      </CardContent>
    </Card>
  );

  const renderPredictions = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Anomaly Predictions
        </Typography>
        <List>
          {predictions.map((prediction) => (
            <ListItem key={`${prediction.provider_id}-${prediction.alert_type}`}>
              <ListItemText
                primary={
                  <Box display="flex" alignItems="center">
                    <Typography variant="subtitle1">
                      {prediction.provider_id} - {prediction.alert_type}
                    </Typography>
                    <Box ml={2}>
                      <Chip
                        label={`${(prediction.probability * 100).toFixed(1)}%`}
                        color={prediction.probability > 0.7 ? 'error' : 'warning'}
                        size="small"
                      />
                    </Box>
                  </Box>
                }
                secondary={
                  <>
                    <Typography variant="body2" color="textSecondary">
                      Predicted at: {new Date(prediction.prediction_time).toLocaleString()}
                    </Typography>
                    <Box mt={1}>
                      <LinearProgress
                        variant="determinate"
                        value={prediction.probability * 100}
                        color={prediction.probability > 0.7 ? 'error' : 'warning'}
                      />
                    </Box>
                  </>
                }
              />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );

  const renderRootCauseAnalysis = () => {
    const data = patterns.flatMap((pattern) =>
      Object.entries(pattern.root_cause_probability).map(([type, prob]) => ({
        pattern: pattern.id,
        type,
        probability: prob * 100,
      }))
    );

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Root Cause Analysis
          </Typography>
          <Box height={300}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" />
                <YAxis unit="%" />
                <Tooltip />
                <Bar dataKey="probability" fill={theme.palette.primary.main} />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>
    );
  };

  return (
    <Box p={3}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          {renderPatternTimeline()}
        </Grid>
        <Grid item xs={12} md={6}>
          {renderPredictions()}
        </Grid>
        <Grid item xs={12}>
          {renderRootCauseAnalysis()}
        </Grid>
      </Grid>
    </Box>
  );
};

export default AlertAnalytics;
