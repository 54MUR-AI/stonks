import React from 'react';
import { Pie } from 'react-chartjs-2';
import { Card, Typography, Box } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import 'chart.js/auto';

const PortfolioComposition = ({ positions }) => {
  const theme = useTheme();

  const data = {
    labels: positions.map(pos => pos.symbol),
    datasets: [
      {
        data: positions.map(pos => pos.weight * 100),
        backgroundColor: [
          theme.palette.primary.main,
          theme.palette.secondary.main,
          theme.palette.error.main,
          theme.palette.warning.main,
          theme.palette.info.main,
          theme.palette.success.main,
          // Generate additional colors for more positions
          ...positions.slice(6).map((_, i) => 
            `hsl(${(i * 137.508) % 360}, 70%, 50%)`
          )
        ],
        borderColor: theme.palette.background.paper,
        borderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          color: theme.palette.text.primary,
          font: {
            family: theme.typography.fontFamily,
          },
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const position = positions[context.dataIndex];
            return [
              `${position.symbol}: ${position.weight.toFixed(2)}%`,
              `Value: $${position.value.toLocaleString()}`,
              `Return: ${position.return.toFixed(2)}%`,
            ];
          },
        },
      },
    },
  };

  return (
    <Card sx={{ p: 2, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Portfolio Composition
      </Typography>
      <Box sx={{ height: 300, position: 'relative' }}>
        <Pie data={data} options={options} />
      </Box>
    </Card>
  );
};

export default PortfolioComposition;
