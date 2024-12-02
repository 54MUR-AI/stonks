import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { useTheme } from '@mui/material/styles';
import { api } from '../../services/api';

interface RiskMetric {
  name: string;
  value: number;
  description: string;
  threshold?: number;
  status: 'normal' | 'warning' | 'danger';
}

interface RiskTrend {
  date: string;
  var_95: number;
  cvar_95: number;
  volatility: number;
  sharpe_ratio: number;
}

interface RiskDecomposition {
  factor: string;
  contribution: number;
  sensitivity: number;
}

interface RiskMetricsProps {
  portfolioId: number;
}

const RiskMetrics: React.FC<RiskMetricsProps> = ({ portfolioId }) => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<RiskMetric[]>([]);
  const [trends, setTrends] = useState<RiskTrend[]>([]);
  const [decomposition, setDecomposition] = useState<RiskDecomposition[]>([]);

  useEffect(() => {
    const fetchRiskData = async () => {
      try {
        const [metricsRes, trendsRes, decompositionRes] = await Promise.all([
          api.get(`/portfolio/${portfolioId}/risk/metrics`),
          api.get(`/portfolio/${portfolioId}/risk/trends`),
          api.get(`/portfolio/${portfolioId}/risk/decomposition`),
        ]);

        setMetrics(metricsRes.data);
        setTrends(trendsRes.data);
        setDecomposition(decompositionRes.data);
      } catch (err) {
        setError('Failed to fetch risk data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchRiskData();
  }, [portfolioId]);

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '400px',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  return (
    <Box>
      {/* Current Risk Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {metrics.map((metric) => (
          <Grid item xs={12} sm={6} md={3} key={metric.name}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  {metric.name}
                </Typography>
                <Typography
                  variant="h5"
                  color={
                    metric.status === 'danger'
                      ? 'error.main'
                      : metric.status === 'warning'
                      ? 'warning.main'
                      : 'success.main'
                  }
                >
                  {metric.value.toFixed(2)}%
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {metric.description}
                </Typography>
                {metric.threshold && (
                  <Typography variant="caption" display="block">
                    Threshold: {metric.threshold}%
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Risk Trends Chart */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Risk Metrics Trends
          </Typography>
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer>
              <LineChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="var_95"
                  stroke={theme.palette.primary.main}
                  name="VaR (95%)"
                />
                <Line
                  type="monotone"
                  dataKey="cvar_95"
                  stroke={theme.palette.secondary.main}
                  name="CVaR (95%)"
                />
                <Line
                  type="monotone"
                  dataKey="volatility"
                  stroke={theme.palette.error.main}
                  name="Volatility"
                />
                <Line
                  type="monotone"
                  dataKey="sharpe_ratio"
                  stroke={theme.palette.success.main}
                  name="Sharpe Ratio"
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>

      {/* Risk Decomposition */}
      <Grid container spacing={3}>
        {/* Radar Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Factor Sensitivities
              </Typography>
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer>
                  <RadarChart cx="50%" cy="50%" outerRadius="80%">
                    <PolarGrid />
                    <PolarAngleAxis dataKey="factor" />
                    <PolarRadiusAxis />
                    <Radar
                      name="Sensitivity"
                      dataKey="sensitivity"
                      data={decomposition}
                      fill={theme.palette.primary.main}
                      fillOpacity={0.6}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Detailed Table */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Factor Contributions
              </Typography>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Risk Factor</TableCell>
                      <TableCell align="right">Contribution (%)</TableCell>
                      <TableCell align="right">Sensitivity</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {decomposition.map((row) => (
                      <TableRow key={row.factor}>
                        <TableCell component="th" scope="row">
                          {row.factor}
                        </TableCell>
                        <TableCell align="right">
                          {(row.contribution * 100).toFixed(2)}%
                        </TableCell>
                        <TableCell align="right">
                          {row.sensitivity.toFixed(3)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RiskMetrics;
