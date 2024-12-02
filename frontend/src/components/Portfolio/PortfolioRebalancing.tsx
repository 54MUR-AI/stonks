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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  BarChart,
  Bar,
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
import { useTheme } from '@mui/material/styles';
import { api } from '../../services/api';

interface PortfolioRebalancingProps {
  portfolioId: number;
}

interface Recommendation {
  symbol: string;
  current_weight: number;
  target_weight: number;
  action: 'buy' | 'sell' | 'hold';
  quantity_change: number;
  expected_impact: {
    return_change: number;
    risk_change: number;
    sharpe_change: number;
    transaction_cost: number;
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const PortfolioRebalancing: React.FC<PortfolioRebalancingProps> = ({ portfolioId }) => {
  const theme = useTheme();
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [objective, setObjective] = useState<string>('sharpe');

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.post(
        `/portfolio/${portfolioId}/rebalance/recommend?objective=${objective}`
      );
      setRecommendations(response.data.recommendations);
    } catch (err) {
      setError('Failed to fetch rebalancing recommendations');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, [portfolioId, objective]);

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

  const getCurrentWeightsData = () =>
    recommendations.map((rec) => ({
      name: rec.symbol,
      value: rec.current_weight,
    }));

  const getTargetWeightsData = () =>
    recommendations.map((rec) => ({
      name: rec.symbol,
      value: rec.target_weight,
    }));

  const getImpactData = () =>
    recommendations.map((rec) => ({
      symbol: rec.symbol,
      'Return Impact': rec.expected_impact.return_change,
      'Risk Impact': -rec.expected_impact.risk_change,
      'Sharpe Ratio Impact': rec.expected_impact.sharpe_change,
    }));

  return (
    <Box>
      <Box mb={3} display="flex" justifyContent="space-between" alignItems="center">
        <FormControl variant="outlined" size="small" style={{ minWidth: 200 }}>
          <InputLabel>Optimization Objective</InputLabel>
          <Select
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            label="Optimization Objective"
          >
            <MenuItem value="sharpe">Maximize Sharpe Ratio</MenuItem>
            <MenuItem value="min_variance">Minimize Variance</MenuItem>
            <MenuItem value="max_diversification">Maximize Diversification</MenuItem>
          </Select>
        </FormControl>
        <Button
          variant="contained"
          color="primary"
          onClick={fetchRecommendations}
        >
          Refresh Recommendations
        </Button>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Portfolio Weights
              </Typography>
              <Box height={300}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={getCurrentWeightsData()}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={(entry) => `${entry.name}: ${(entry.value * 100).toFixed(1)}%`}
                    >
                      {getCurrentWeightsData().map((entry, index) => (
                        <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Target Portfolio Weights
              </Typography>
              <Box height={300}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={getTargetWeightsData()}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={(entry) => `${entry.name}: ${(entry.value * 100).toFixed(1)}%`}
                    >
                      {getTargetWeightsData().map((entry, index) => (
                        <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Expected Impact
              </Typography>
              <Box height={300}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getImpactData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="symbol" />
                    <YAxis tickFormatter={(value) => `${(value * 100).toFixed(1)}%`} />
                    <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
                    <Legend />
                    <Bar dataKey="Return Impact" fill="#4CAF50" />
                    <Bar dataKey="Risk Impact" fill="#f44336" />
                    <Bar dataKey="Sharpe Ratio Impact" fill="#2196F3" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell align="right">Current Weight</TableCell>
                  <TableCell align="right">Target Weight</TableCell>
                  <TableCell align="right">Action</TableCell>
                  <TableCell align="right">Quantity Change</TableCell>
                  <TableCell align="right">Transaction Cost</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {recommendations.map((rec) => (
                  <TableRow
                    key={rec.symbol}
                    sx={{
                      backgroundColor:
                        rec.action === 'buy'
                          ? alpha(theme.palette.success.main, 0.1)
                          : rec.action === 'sell'
                          ? alpha(theme.palette.error.main, 0.1)
                          : 'inherit',
                    }}
                  >
                    <TableCell>{rec.symbol}</TableCell>
                    <TableCell align="right">
                      {(rec.current_weight * 100).toFixed(2)}%
                    </TableCell>
                    <TableCell align="right">
                      {(rec.target_weight * 100).toFixed(2)}%
                    </TableCell>
                    <TableCell align="right">
                      <Typography
                        color={
                          rec.action === 'buy'
                            ? 'success.main'
                            : rec.action === 'sell'
                            ? 'error.main'
                            : 'text.primary'
                        }
                      >
                        {rec.action.toUpperCase()}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">{rec.quantity_change}</TableCell>
                    <TableCell align="right">
                      ${rec.expected_impact.transaction_cost.toFixed(2)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PortfolioRebalancing;
