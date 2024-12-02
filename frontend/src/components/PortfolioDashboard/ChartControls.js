import React from 'react';
import {
  Box,
  ToggleButton,
  ToggleButtonGroup,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
} from '@mui/material';

const timeRanges = [
  { value: '1D', label: '1D' },
  { value: '1W', label: '1W' },
  { value: '1M', label: '1M' },
  { value: '3M', label: '3M' },
  { value: '6M', label: '6M' },
  { value: '1Y', label: '1Y' },
  { value: 'ALL', label: 'ALL' },
];

const indicators = [
  { value: 'SMA', label: 'Simple Moving Average' },
  { value: 'EMA', label: 'Exponential Moving Average' },
  { value: 'BB', label: 'Bollinger Bands' },
  { value: 'RSI', label: 'Relative Strength Index' },
  { value: 'MACD', label: 'MACD' },
];

const benchmarks = [
  { value: 'SPY', label: 'S&P 500 (SPY)' },
  { value: 'QQQ', label: 'NASDAQ (QQQ)' },
  { value: 'DIA', label: 'Dow Jones (DIA)' },
  { value: 'IWM', label: 'Russell 2000 (IWM)' },
];

const ChartControls = ({
  timeRange,
  selectedIndicators,
  benchmark,
  onTimeRangeChange,
  onIndicatorChange,
  onBenchmarkChange,
}) => {
  return (
    <Box sx={{ mb: 2 }}>
      <Stack direction="row" spacing={2} alignItems="center">
        {/* Time Range Selector */}
        <ToggleButtonGroup
          value={timeRange}
          exclusive
          onChange={(e, value) => value && onTimeRangeChange(value)}
          size="small"
        >
          {timeRanges.map(({ value, label }) => (
            <ToggleButton key={value} value={value}>
              {label}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>

        {/* Technical Indicators */}
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel>Indicators</InputLabel>
          <Select
            multiple
            value={selectedIndicators}
            onChange={(e) => onIndicatorChange(e.target.value)}
            label="Indicators"
            renderValue={(selected) => 
              selected
                .map(value => indicators.find(i => i.value === value)?.label)
                .join(', ')
            }
          >
            {indicators.map(({ value, label }) => (
              <MenuItem key={value} value={value}>
                {label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Benchmark Selector */}
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Benchmark</InputLabel>
          <Select
            value={benchmark}
            onChange={(e) => onBenchmarkChange(e.target.value)}
            label="Benchmark"
          >
            <MenuItem value="">None</MenuItem>
            {benchmarks.map(({ value, label }) => (
              <MenuItem key={value} value={value}>
                {label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Stack>
    </Box>
  );
};

export default ChartControls;
