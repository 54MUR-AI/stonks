import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Chip,
} from '@mui/material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

interface RiskMetrics {
  volatility: number;
  value_at_risk: {
    historical: number;
    parametric: number;
    modified: number;
  };
  expected_shortfall: number;
  beta: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  risk_contribution: Record<string, number>;
  stress_tests?: {
    [key: string]: {
      description: string;
      impact: number;
      max_drawdown: number;
      recovery_days: number;
    };
  };
}

const COLORS = [
  '#0088FE',
  '#00C49F',
  '#FFBB28',
  '#FF8042',
  '#8884D8',
  '#82CA9D',
];

const RiskDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<number>(0);

  useEffect(() => {
    fetchRiskMetrics();
  }, []);

  const fetchRiskMetrics = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/risk/metrics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          positions: await getCurrentPositions(),
          include_stress_tests: true,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch risk metrics');
      }

      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getCurrentPositions = async () => {
    // TODO: Implement fetching current portfolio positions
    return {};
  };

  const renderKeyMetrics = () => {
    if (!metrics) return null;

    const keyMetrics = [
      {
        label: 'Volatility',
        value: (metrics.volatility * 100).toFixed(2) + '%',
        color: metrics.volatility > 0.2 ? 'error' : 'success',
      },
      {
        label: 'VaR (95%)',
        value: (metrics.value_at_risk.historical * 100).toFixed(2) + '%',
        color: metrics.value_at_risk.historical > 0.1 ? 'error' : 'success',
      },
      {
        label: 'Expected Shortfall',
        value: (metrics.expected_shortfall * 100).toFixed(2) + '%',
        color: metrics.expected_shortfall > 0.15 ? 'error' : 'success',
      },
      {
        label: 'Beta',
        value: metrics.beta.toFixed(2),
        color: Math.abs(metrics.beta - 1) > 0.3 ? 'warning' : 'success',
      },
      {
        label: 'Sharpe Ratio',
        value: metrics.sharpe_ratio.toFixed(2),
        color: metrics.sharpe_ratio < 1 ? 'warning' : 'success',
      },
      {
        label: 'Sortino Ratio',
        value: metrics.sortino_ratio.toFixed(2),
        color: metrics.sortino_ratio < 1 ? 'warning' : 'success',
      },
    ];

    return (
      <Grid container spacing={2}>
        {keyMetrics.map((metric, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card>
              <CardContent>
                <Typography variant="subtitle2" color="textSecondary">
                  {metric.label}
                </Typography>
                <Box display="flex" alignItems="center" mt={1}>
                  <Typography variant="h4" component="div">
                    {metric.value}
                  </Typography>
                  <Box ml={1}>
                    <Chip
                      size="small"
                      color={metric.color as any}
                      label={metric.color.toUpperCase()}
                    />
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  };

  const renderRiskContribution = () => {
    if (!metrics) return null;

    const data = Object.entries(metrics.risk_contribution).map(([symbol, value]) => ({
      symbol,
      contribution: value * 100,
    }));

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Risk Contribution
          </Typography>
          <Box height={300}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={data}
                  dataKey="contribution"
                  nameKey="symbol"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  label
                >
                  {data.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => `${value.toFixed(2)}%`} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>
    );
  };

  const renderStressTests = () => {
    if (!metrics?.stress_tests) return null;

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Stress Test Scenarios
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Scenario</TableCell>
                  <TableCell>Description</TableCell>
                  <TableCell align="right">Impact</TableCell>
                  <TableCell align="right">Max Drawdown</TableCell>
                  <TableCell align="right">Recovery Days</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(metrics.stress_tests).map(([key, test]) => (
                  <TableRow key={key}>
                    <TableCell component="th" scope="row">
                      {key.replace(/_/g, ' ')}
                    </TableCell>
                    <TableCell>{test.description}</TableCell>
                    <TableCell align="right">
                      {(test.impact * 100).toFixed(2)}%
                    </TableCell>
                    <TableCell align="right">
                      {(test.max_drawdown * 100).toFixed(2)}%
                    </TableCell>
                    <TableCell align="right">{test.recovery_days}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    );
  };

  const renderVaRComparison = () => {
    if (!metrics) return null;

    const data = [
      {
        name: 'Historical',
        value: metrics.value_at_risk.historical * 100,
      },
      {
        name: 'Parametric',
        value: metrics.value_at_risk.parametric * 100,
      },
      {
        name: 'Modified',
        value: metrics.value_at_risk.modified * 100,
      },
    ];

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            VaR Comparison
          </Typography>
          <Box height={300}>
            <ResponsiveContainer>
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis tickFormatter={(value) => `${value.toFixed(2)}%`} />
                <Tooltip
                  formatter={(value: number) => [`${value.toFixed(2)}%`, 'VaR']}
                />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>
        Risk Analytics Dashboard
      </Typography>

      <Box mb={3}>
        <Tabs
          value={activeTab}
          onChange={(_, newValue) => setActiveTab(newValue)}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="Overview" />
          <Tab label="Risk Contribution" />
          <Tab label="Stress Tests" />
          <Tab label="VaR Analysis" />
        </Tabs>
      </Box>

      {activeTab === 0 && (
        <Box>
          {renderKeyMetrics()}
        </Box>
      )}

      {activeTab === 1 && (
        <Box>
          {renderRiskContribution()}
        </Box>
      )}

      {activeTab === 2 && (
        <Box>
          {renderStressTests()}
        </Box>
      )}

      {activeTab === 3 && (
        <Box>
          {renderVaRComparison()}
        </Box>
      )}
    </Box>
  );
};

export default RiskDashboard;
