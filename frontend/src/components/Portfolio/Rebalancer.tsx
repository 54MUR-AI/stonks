import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  TextField,
  Grid,
  Slider,
  FormControlLabel,
  Switch,
  Chip,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from 'recharts';

interface PortfolioHolding {
  symbol: string;
  quantity: number;
  target_weight: number;
  current_weight?: number;
  price?: number;
  market_value?: number;
}

interface RebalanceAction {
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  estimated_value: number;
  reason: string;
}

interface RebalanceResult {
  actions: RebalanceAction[];
  total_value: number;
  current_weights: Record<string, number>;
  target_weights: Record<string, number>;
  tracking_error: number;
  estimated_turnover: number;
  risk_impact: Record<string, number>;
}

const COLORS = [
  '#0088FE',
  '#00C49F',
  '#FFBB28',
  '#FF8042',
  '#8884D8',
  '#82CA9D',
  '#FFC658',
];

const Rebalancer: React.FC = () => {
  const [holdings, setHoldings] = useState<PortfolioHolding[]>([]);
  const [cash, setCash] = useState<number>(0);
  const [rebalanceThreshold, setRebalanceThreshold] = useState<number>(2);
  const [minTradeValue, setMinTradeValue] = useState<number>(100);
  const [maxTurnover, setMaxTurnover] = useState<number>(20);
  const [optimizeTracking, setOptimizeTracking] = useState<boolean>(false);
  const [optimizeTurnover, setOptimizeTurnover] = useState<boolean>(false);
  const [result, setResult] = useState<RebalanceResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const fetchPortfolio = async () => {
    try {
      const response = await fetch('/api/portfolio/holdings');
      const data = await response.json();
      setHoldings(data);
    } catch (err) {
      setError('Failed to fetch portfolio data');
    }
  };

  const handleRebalance = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/portfolio/rebalance', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          holdings,
          cash,
          constraints: {
            minimize_tracking_error: optimizeTracking,
            minimize_turnover: optimizeTurnover,
          },
          rebalance_threshold: rebalanceThreshold / 100,
          min_trade_value: minTradeValue,
          max_turnover: maxTurnover / 100,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Rebalancing calculation failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const renderHoldingsTable = () => (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Quantity</TableCell>
            <TableCell align="right">Current Weight</TableCell>
            <TableCell align="right">Target Weight</TableCell>
            <TableCell align="right">Market Value</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {holdings.map((holding) => (
            <TableRow key={holding.symbol}>
              <TableCell>{holding.symbol}</TableCell>
              <TableCell align="right">{holding.quantity.toFixed(6)}</TableCell>
              <TableCell align="right">
                {(holding.current_weight || 0).toLocaleString(undefined, {
                  style: 'percent',
                  minimumFractionDigits: 2,
                })}
              </TableCell>
              <TableCell align="right">
                {holding.target_weight.toLocaleString(undefined, {
                  style: 'percent',
                  minimumFractionDigits: 2,
                })}
              </TableCell>
              <TableCell align="right">
                {holding.market_value?.toLocaleString(undefined, {
                  style: 'currency',
                  currency: 'USD',
                })}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );

  const renderWeightChart = () => {
    if (!result) return null;

    const data = Object.keys(result.current_weights).map((symbol) => ({
      symbol,
      current: result.current_weights[symbol],
      target: result.target_weights[symbol],
    }));

    return (
      <Box height={300}>
        <ResponsiveContainer>
          <BarChart data={data}>
            <XAxis dataKey="symbol" />
            <YAxis tickFormatter={(value) => `${(value * 100).toFixed(1)}%`} />
            <Tooltip
              formatter={(value: number) =>
                `${(value * 100).toFixed(2)}%`
              }
            />
            <Legend />
            <Bar
              name="Current Weight"
              dataKey="current"
              fill="#8884d8"
              opacity={0.8}
            />
            <Bar
              name="Target Weight"
              dataKey="target"
              fill="#82ca9d"
              opacity={0.8}
            />
          </BarChart>
        </ResponsiveContainer>
      </Box>
    );
  };

  const renderActions = () => {
    if (!result?.actions.length) return null;

    return (
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell>Action</TableCell>
              <TableCell align="right">Quantity</TableCell>
              <TableCell align="right">Value</TableCell>
              <TableCell>Reason</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {result.actions.map((action, index) => (
              <TableRow key={index}>
                <TableCell>{action.symbol}</TableCell>
                <TableCell>
                  <Chip
                    label={action.action}
                    color={action.action === 'BUY' ? 'success' : 'error'}
                    size="small"
                  />
                </TableCell>
                <TableCell align="right">{action.quantity.toFixed(6)}</TableCell>
                <TableCell align="right">
                  {action.estimated_value.toLocaleString(undefined, {
                    style: 'currency',
                    currency: 'USD',
                  })}
                </TableCell>
                <TableCell>{action.reason}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  const renderMetrics = () => {
    if (!result) return null;

    return (
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Portfolio Value
              </Typography>
              <Typography variant="h4">
                {result.total_value.toLocaleString(undefined, {
                  style: 'currency',
                  currency: 'USD',
                })}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Turnover
              </Typography>
              <Typography variant="h4">
                {(result.estimated_turnover * 100).toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Tracking Error
              </Typography>
              <Typography variant="h4">
                {(result.tracking_error * 100).toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>
        Portfolio Rebalancer
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Holdings
              </Typography>
              {renderHoldingsTable()}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Rebalancing Parameters
              </Typography>
              <Box mb={2}>
                <TextField
                  fullWidth
                  label="Cash Available"
                  type="number"
                  value={cash}
                  onChange={(e) => setCash(Number(e.target.value))}
                  InputProps={{
                    startAdornment: '$',
                  }}
                />
              </Box>
              <Box mb={2}>
                <Typography gutterBottom>
                  Rebalance Threshold: {rebalanceThreshold}%
                </Typography>
                <Slider
                  value={rebalanceThreshold}
                  onChange={(_, value) => setRebalanceThreshold(value as number)}
                  min={0}
                  max={10}
                  step={0.5}
                />
              </Box>
              <Box mb={2}>
                <Typography gutterBottom>
                  Maximum Turnover: {maxTurnover}%
                </Typography>
                <Slider
                  value={maxTurnover}
                  onChange={(_, value) => setMaxTurnover(value as number)}
                  min={0}
                  max={100}
                  step={5}
                />
              </Box>
              <Box mb={2}>
                <TextField
                  fullWidth
                  label="Minimum Trade Value"
                  type="number"
                  value={minTradeValue}
                  onChange={(e) => setMinTradeValue(Number(e.target.value))}
                  InputProps={{
                    startAdornment: '$',
                  }}
                />
              </Box>
              <Box mb={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={optimizeTracking}
                      onChange={(e) => setOptimizeTracking(e.target.checked)}
                    />
                  }
                  label="Minimize Tracking Error"
                />
              </Box>
              <Box mb={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={optimizeTurnover}
                      onChange={(e) => setOptimizeTurnover(e.target.checked)}
                    />
                  }
                  label="Minimize Turnover"
                />
              </Box>
              <Button
                fullWidth
                variant="contained"
                color="primary"
                onClick={handleRebalance}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Calculate Rebalance'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {result && (
          <>
            <Grid item xs={12}>
              {renderMetrics()}
            </Grid>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Weight Comparison
                  </Typography>
                  {renderWeightChart()}
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Rebalancing Actions
                  </Typography>
                  {renderActions()}
                </CardContent>
              </Card>
            </Grid>
          </>
        )}
      </Grid>
    </Box>
  );
};

export default Rebalancer;
