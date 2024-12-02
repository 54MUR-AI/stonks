import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Chip,
  CircularProgress,
  Alert,
  Paper,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  Scatter,
  ScatterChart,
} from 'recharts';
import { api } from '../../services/api';

interface ScenarioType {
  name: string;
  type: 'historical' | 'hypothetical' | 'monte_carlo' | 'sensitivity' | 'regime_change';
  description: string;
  probability?: number;
}

interface StressTestResult {
  scenario_name: string;
  portfolio_impact: number;
  asset_impacts: Record<string, number>;
  risk_metrics: {
    stressed_var_95: number;
    stressed_sharpe: number;
    portfolio_volatility: number;
  };
  correlation_changes?: Record<string, number>;
  volatility_changes?: Record<string, number>;
}

const StressTestDashboard: React.FC<{ portfolioId: number }> = ({ portfolioId }) => {
  const theme = useTheme();
  const [scenarios, setScenarios] = useState<ScenarioType[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<string>('');
  const [selectedType, setSelectedType] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<StressTestResult | null>(null);

  // Fetch available scenarios
  useEffect(() => {
    const fetchScenarios = async () => {
      try {
        const response = await api.get(
          `/portfolio/${portfolioId}/stress-test/scenarios${selectedType ? `?scenario_type=${selectedType}` : ''}`
        );
        setScenarios(response.data.scenarios);
      } catch (err) {
        setError('Failed to fetch scenarios');
        console.error(err);
      }
    };

    fetchScenarios();
  }, [portfolioId, selectedType]);

  const runStressTest = async () => {
    if (!selectedScenario) return;

    setLoading(true);
    setError(null);
    try {
      const response = await api.post(`/portfolio/${portfolioId}/stress-test/run`, {
        scenario_name: selectedScenario,
        scenario_type: selectedType,
      });
      setResults(response.data);
    } catch (err) {
      setError('Failed to run stress test');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Format data for charts
  const getImpactChartData = () => {
    if (!results) return [];
    return Object.entries(results.asset_impacts).map(([symbol, impact]) => ({
      symbol,
      impact: impact * 100, // Convert to percentage
    }));
  };

  const getRiskMetricsData = () => {
    if (!results) return [];
    return [
      {
        name: 'VaR (95%)',
        value: results.risk_metrics.stressed_var_95 * 100,
      },
      {
        name: 'Portfolio Volatility',
        value: results.risk_metrics.portfolio_volatility * 100,
      },
      {
        name: 'Sharpe Ratio',
        value: results.risk_metrics.stressed_sharpe,
      },
    ];
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Portfolio Stress Testing
      </Typography>

      {/* Controls */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Scenario Type</InputLabel>
              <Select
                value={selectedType}
                label="Scenario Type"
                onChange={(e) => setSelectedType(e.target.value)}
              >
                <MenuItem value="">All Types</MenuItem>
                <MenuItem value="historical">Historical</MenuItem>
                <MenuItem value="monte_carlo">Monte Carlo</MenuItem>
                <MenuItem value="sensitivity">Sensitivity</MenuItem>
                <MenuItem value="regime_change">Regime Change</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Scenario</InputLabel>
              <Select
                value={selectedScenario}
                label="Scenario"
                onChange={(e) => setSelectedScenario(e.target.value)}
              >
                {scenarios.map((scenario) => (
                  <MenuItem key={scenario.name} value={scenario.name}>
                    {scenario.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <Button
              variant="contained"
              onClick={runStressTest}
              disabled={!selectedScenario || loading}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : 'Run Stress Test'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Results Display */}
      {results && (
        <Grid container spacing={3}>
          {/* Overall Impact */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Portfolio Impact
                </Typography>
                <Typography variant="h3" color={results.portfolio_impact < 0 ? 'error' : 'success'}>
                  {(results.portfolio_impact * 100).toFixed(2)}%
                </Typography>
                <Typography variant="subtitle2" color="textSecondary">
                  Under scenario: {results.scenario_name}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Risk Metrics */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Stressed Risk Metrics
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  {Object.entries(results.risk_metrics).map(([key, value]) => (
                    <Chip
                      key={key}
                      label={`${key}: ${(value * 100).toFixed(2)}%`}
                      color="primary"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Asset Impact Chart */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Asset Impacts
                </Typography>
                <Box sx={{ height: 400 }}>
                  <ResponsiveContainer>
                    <BarChart data={getImpactChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="symbol" />
                      <YAxis label={{ value: 'Impact (%)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Bar
                        dataKey="impact"
                        fill={theme.palette.primary.main}
                        name="Impact (%)"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Risk Metrics Chart */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Risk Metrics Comparison
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer>
                    <BarChart data={getRiskMetricsData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar
                        dataKey="value"
                        fill={theme.palette.secondary.main}
                        name="Value"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Correlation Changes */}
          {results.correlation_changes && (
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Correlation Changes
                  </Typography>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" dataKey="x" name="Original" />
                        <YAxis type="number" dataKey="y" name="Stressed" />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter
                          name="Correlations"
                          data={Object.entries(results.correlation_changes).map(([pair, change]) => ({
                            x: 0,
                            y: change,
                            pair,
                          }))}
                          fill={theme.palette.info.main}
                        />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}
    </Box>
  );
};

export default StressTestDashboard;
