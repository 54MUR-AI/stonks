import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import axios from 'axios';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  TextField, 
  Button,
  Box,
  Paper,
  Grid,
  Card,
  CardContent
} from '@mui/material';

function App() {
  const chartContainerRef = useRef();
  const [symbol, setSymbol] = useState('AAPL');
  const [analysis, setAnalysis] = useState('');
  const [chart, setChart] = useState(null);

  useEffect(() => {
    // Create chart instance
    const chartInstance = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#1E222D' },
        textColor: '#DDD',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
    });

    // Create candlestick series
    const candleSeries = chartInstance.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    setChart({ instance: chartInstance, series: candleSeries });

    // Fetch initial data
    fetchMarketData(symbol, candleSeries);

    // Cleanup
    return () => {
      chartInstance.remove();
    };
  }, []);

  const fetchMarketData = async (sym, series) => {
    try {
      const response = await axios.get(`http://localhost:8000/market/${sym}`);
      series.setData(response.data.data);
      
      // Get AI analysis
      const aiResponse = await axios.get(`http://localhost:8000/ai/analysis/${sym}`);
      setAnalysis(aiResponse.data.analysis);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleSymbolChange = (event) => {
    const newSymbol = event.target.value.toUpperCase();
    setSymbol(newSymbol);
  };

  const handleSubmit = () => {
    if (chart) {
      fetchMarketData(symbol, chart.series);
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" sx={{ backgroundColor: '#1E222D' }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Stonks
          </Typography>
          <TextField
            value={symbol}
            onChange={handleSymbolChange}
            variant="outlined"
            size="small"
            sx={{ 
              mr: 2,
              input: { color: 'white' },
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'white' },
              }
            }}
          />
          <Button 
            variant="contained" 
            onClick={handleSubmit}
            sx={{ backgroundColor: '#2962FF' }}
          >
            Search
          </Button>
        </Toolbar>
      </AppBar>

      <Grid container spacing={2} sx={{ p: 2 }}>
        <Grid item xs={12} md={9}>
          <Paper 
            ref={chartContainerRef} 
            sx={{ 
              p: 2,
              backgroundColor: '#1E222D',
              height: '500px'
            }}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#1E222D', color: 'white' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AI Analysis
              </Typography>
              <Typography variant="body2">
                {analysis || 'Loading analysis...'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default App;
