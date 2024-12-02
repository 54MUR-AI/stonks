import React from 'react';
import { Scatter } from 'react-chartjs-2';
import { Card, Typography, Box } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import 'chart.js/auto';

const RiskReturnPlot = ({ positions }) => {
  const theme = useTheme();

  const data = {
    datasets: [{
      label: 'Risk/Return Profile',
      data: positions.map(pos => ({
        x: pos.volatility,
        y: pos.return,
        symbol: pos.symbol,
        weight: pos.weight,
        value: pos.value,
      })),
      backgroundColor: theme.palette.primary.main,
      borderColor: theme.palette.primary.dark,
      pointRadius: positions.map(pos => Math.max(5, pos.weight * 20)), // Size based on position weight
      pointHoverRadius: positions.map(pos => Math.max(7, pos.weight * 25)),
    }],
  };

  const options = {
    responsive: true,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Volatility (%)',
          color: theme.palette.text.primary,
        },
        grid: {
          color: theme.palette.divider,
        },
        ticks: {
          color: theme.palette.text.secondary,
        },
      },
      y: {
        title: {
          display: true,
          text: 'Return (%)',
          color: theme.palette.text.primary,
        },
        grid: {
          color: theme.palette.divider,
        },
        ticks: {
          color: theme.palette.text.secondary,
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const point = context.raw;
            return [
              `Symbol: ${point.symbol}`,
              `Return: ${point.y.toFixed(2)}%`,
              `Volatility: ${point.x.toFixed(2)}%`,
              `Weight: ${(point.weight * 100).toFixed(2)}%`,
              `Value: $${point.value.toLocaleString()}`,
            ];
          },
        },
      },
    },
  };

  return (
    <Card sx={{ p: 2, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Risk/Return Analysis
      </Typography>
      <Box sx={{ height: 300, position: 'relative' }}>
        <Scatter data={data} options={options} />
      </Box>
    </Card>
  );
};

export default RiskReturnPlot;
