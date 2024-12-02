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
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { useTheme } from '@mui/material/styles';
import { api } from '../../services/api';

interface RiskPredictionProps {
  portfolioId: number;
}

interface ModelPrediction {
  random_forest: number;
  gradient_boosting: number;
  lstm: number;
}

interface RiskPrediction {
  symbol: string;
  predicted_volatility: number;
  model_predictions: ModelPrediction;
  confidence_interval: [number, number];
}

const RiskPrediction: React.FC<RiskPredictionProps> = ({ portfolioId }) => {
  const theme = useTheme();
  const [predictions, setPredictions] = useState<RiskPrediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [daysForward, setDaysForward] = useState(5);
  const [isTraining, setIsTraining] = useState(false);

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.post(
        `/portfolio/${portfolioId}/risk/predict?days_forward=${daysForward}`
      );
      const formattedPredictions = Object.entries(response.data.predictions.risk_predictions).map(
        ([symbol, value]) => ({
          symbol,
          predicted_volatility: value,
          model_predictions: response.data.predictions.model_predictions[symbol],
          confidence_interval: response.data.predictions.confidence_intervals[symbol],
        })
      );
      setPredictions(formattedPredictions);
    } catch (err) {
      setError('Failed to fetch risk predictions');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const trainModels = async () => {
    setIsTraining(true);
    setError(null);
    try {
      await api.post(`/portfolio/${portfolioId}/risk/train`);
      await fetchPredictions();
    } catch (err) {
      setError('Failed to train risk models');
      console.error(err);
    } finally {
      setIsTraining(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, [portfolioId, daysForward]);

  if (loading || isTraining) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  const getModelComparisonData = (prediction: RiskPrediction) => [
    {
      name: 'Random Forest',
      value: prediction.model_predictions.random_forest,
    },
    {
      name: 'Gradient Boosting',
      value: prediction.model_predictions.gradient_boosting,
    },
    {
      name: 'LSTM',
      value: prediction.model_predictions.lstm,
    },
  ];

  return (
    <Box>
      <Box mb={3} display="flex" justifyContent="space-between" alignItems="center">
        <FormControl variant="outlined" size="small" style={{ minWidth: 120 }}>
          <InputLabel>Days Forward</InputLabel>
          <Select
            value={daysForward}
            onChange={(e) => setDaysForward(Number(e.target.value))}
            label="Days Forward"
          >
            <MenuItem value={5}>5 Days</MenuItem>
            <MenuItem value={10}>10 Days</MenuItem>
            <MenuItem value={20}>20 Days</MenuItem>
            <MenuItem value={30}>30 Days</MenuItem>
          </Select>
        </FormControl>
        <Button
          variant="contained"
          color="primary"
          onClick={trainModels}
          disabled={isTraining}
        >
          {isTraining ? 'Training...' : 'Train Models'}
        </Button>
      </Box>

      <Grid container spacing={3}>
        {predictions.map((prediction) => (
          <Grid item xs={12} md={6} key={prediction.symbol}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {prediction.symbol} Risk Prediction
                </Typography>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Predicted Volatility: {(prediction.predicted_volatility * 100).toFixed(2)}%
                  <br />
                  Confidence Interval: {(prediction.confidence_interval[0] * 100).toFixed(2)}% -{' '}
                  {(prediction.confidence_interval[1] * 100).toFixed(2)}%
                </Typography>

                <Box height={300}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getModelComparisonData(prediction)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis
                        tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                      />
                      <Tooltip
                        formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                      />
                      <Bar
                        dataKey="value"
                        fill={theme.palette.primary.main}
                        name="Predicted Volatility"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default RiskPrediction;
