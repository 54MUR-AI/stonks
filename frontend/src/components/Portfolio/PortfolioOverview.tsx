import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { useTheme } from '@mui/material/styles';

interface Position {
  symbol: string;
  quantity: number;
  average_price: number;
  current_price: number;
  value: number;
  weight: number;
  return: number;
}

interface Portfolio {
  id: number;
  name: string;
  description: string;
  total_value: number;
  cash: number;
  positions: Position[];
  historical_values: Array<{
    date: string;
    value: number;
  }>;
  total_return: number;
  daily_return: number;
}

interface PortfolioOverviewProps {
  portfolio: Portfolio;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const PortfolioOverview: React.FC<PortfolioOverviewProps> = ({ portfolio }) => {
  const theme = useTheme();

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <Box>
      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Value
              </Typography>
              <Typography variant="h5">
                {formatCurrency(portfolio.total_value)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Cash Balance
              </Typography>
              <Typography variant="h5">
                {formatCurrency(portfolio.cash)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Return
              </Typography>
              <Typography
                variant="h5"
                color={portfolio.total_return >= 0 ? 'success.main' : 'error.main'}
              >
                {formatPercentage(portfolio.total_return)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Daily Return
              </Typography>
              <Typography
                variant="h5"
                color={portfolio.daily_return >= 0 ? 'success.main' : 'error.main'}
              >
                {formatPercentage(portfolio.daily_return)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Portfolio Value Chart */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Portfolio Value History
          </Typography>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer>
              <AreaChart
                data={portfolio.historical_values}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={theme.palette.primary.main}
                  fill={theme.palette.primary.light}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>

      {/* Holdings */}
      <Grid container spacing={3}>
        {/* Asset Allocation */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Asset Allocation
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie
                      data={portfolio.positions}
                      dataKey="value"
                      nameKey="symbol"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label
                    >
                      {portfolio.positions.map((entry, index) => (
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
            </CardContent>
          </Card>
        </Grid>

        {/* Positions List */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Positions
              </Typography>
              {portfolio.positions.map((position) => (
                <Box
                  key={position.symbol}
                  sx={{
                    mb: 2,
                    p: 2,
                    border: 1,
                    borderColor: 'divider',
                    borderRadius: 1,
                  }}
                >
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      mb: 1,
                    }}
                  >
                    <Typography variant="subtitle1">{position.symbol}</Typography>
                    <Chip
                      label={formatPercentage(position.return)}
                      color={position.return >= 0 ? 'success' : 'error'}
                      size="small"
                    />
                  </Box>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Quantity
                      </Typography>
                      <Typography>{position.quantity}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Value
                      </Typography>
                      <Typography>{formatCurrency(position.value)}</Typography>
                    </Grid>
                  </Grid>
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2" color="textSecondary">
                      Portfolio Weight
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={position.weight * 100}
                      sx={{ mt: 0.5 }}
                    />
                  </Box>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PortfolioOverview;
