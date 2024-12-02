import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import { Box, Card, Grid, Typography, CircularProgress, Alert } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import ChartControls from './ChartControls';
import PortfolioComposition from './PortfolioComposition';
import RiskReturnPlot from './RiskReturnPlot';

const PortfolioDashboard = ({ portfolioId }) => {
  const theme = useTheme();
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const wsRef = useRef(null);
  const [portfolioData, setPortfolioData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [wsStatus, setWsStatus] = useState('connecting');
  
  // Chart control states
  const [timeRange, setTimeRange] = useState('1M');
  const [selectedIndicators, setSelectedIndicators] = useState([]);
  const [benchmark, setBenchmark] = useState('');

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(`ws://localhost:8000/api/ws/portfolio/${portfolioId}`);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setWsStatus('connected');
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'price_update') {
          // Update portfolio data with new price
          setPortfolioData(prevData => {
            if (!prevData) return null;
            
            const updatedPositions = prevData.positions.map(pos => {
              if (pos.symbol === data.symbol) {
                const newValue = pos.quantity * data.price;
                return { ...pos, currentPrice: data.price, value: newValue };
              }
              return pos;
            });

            // Calculate new total value
            const newTotalValue = updatedPositions.reduce((sum, pos) => sum + pos.value, 0);
            
            // Update chart with new value
            if (chartRef.current && chartRef.current.mainSeries) {
              chartRef.current.mainSeries.update({
                time: data.timestamp,
                value: newTotalValue
              });
            }

            return {
              ...prevData,
              currentValue: newTotalValue,
              positions: updatedPositions
            };
          });
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setWsStatus('disconnected');
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsStatus('error');
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [portfolioId]);

  useEffect(() => {
    const fetchPortfolioData = async () => {
      try {
        const response = await fetch(`/api/portfolios/${portfolioId}/metrics?timeRange=${timeRange}`);
        if (!response.ok) throw new Error('Failed to fetch portfolio data');
        const data = await response.json();
        setPortfolioData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolioData();
  }, [portfolioId, timeRange]);

  useEffect(() => {
    if (!chartContainerRef.current || !portfolioData) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: theme.palette.background.paper },
        textColor: theme.palette.text.primary,
      },
      grid: {
        vertLines: { color: theme.palette.divider },
        horzLines: { color: theme.palette.divider },
      },
      rightPriceScale: {
        borderColor: theme.palette.divider,
      },
      timeScale: {
        borderColor: theme.palette.divider,
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Main price series
    const mainSeries = chart.addLineSeries({
      color: theme.palette.primary.main,
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: true,
      crosshairMarkerVisible: true,
    });

    // Add benchmark if selected
    let benchmarkSeries = null;
    if (benchmark && portfolioData.benchmarkData) {
      benchmarkSeries = chart.addLineSeries({
        color: theme.palette.secondary.main,
        lineWidth: 1,
        lastValueVisible: true,
        priceLineVisible: false,
      });
      benchmarkSeries.setData(portfolioData.benchmarkData);
    }

    // Add technical indicators
    const indicatorSeries = {};
    selectedIndicators.forEach(indicator => {
      if (portfolioData.indicators && portfolioData.indicators[indicator]) {
        switch (indicator) {
          case 'SMA':
          case 'EMA':
            indicatorSeries[indicator] = chart.addLineSeries({
              color: indicator === 'SMA' ? theme.palette.success.main : theme.palette.warning.main,
              lineWidth: 1,
              lastValueVisible: true,
            });
            indicatorSeries[indicator].setData(portfolioData.indicators[indicator]);
            break;
          case 'BB':
            // Add upper and lower bands
            indicatorSeries.bbUpper = chart.addLineSeries({
              color: theme.palette.info.main,
              lineWidth: 1,
              lastValueVisible: true,
            });
            indicatorSeries.bbLower = chart.addLineSeries({
              color: theme.palette.info.main,
              lineWidth: 1,
              lastValueVisible: true,
            });
            indicatorSeries.bbUpper.setData(portfolioData.indicators.BB.upper);
            indicatorSeries.bbLower.setData(portfolioData.indicators.BB.lower);
            break;
          // Add more indicators as needed
        }
      }
    });

    // Set main series data
    mainSeries.setData(portfolioData.historicalValue);

    // Store references for real-time updates
    chartRef.current = {
      chart,
      mainSeries,
      benchmarkSeries,
      indicatorSeries,
    };

    const handleResize = () => {
      chart.applyOptions({
        width: chartContainerRef.current.clientWidth,
      });
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [portfolioData, theme, selectedIndicators, benchmark]);

  if (loading) return <CircularProgress />;
  if (error) return <Typography color="error">{error}</Typography>;
  if (!portfolioData) return null;

  return (
    <Box sx={{ p: 3 }}>
      {wsStatus === 'error' && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Connection error. Attempting to reconnect...
        </Alert>
      )}
      
      {wsStatus === 'disconnected' && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Disconnected from server. Reconnecting...
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Portfolio Value Card */}
        <Grid item xs={12} md={4}>
          <Card sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Portfolio Value
            </Typography>
            <Typography variant="h4">
              ${portfolioData.currentValue.toLocaleString()}
            </Typography>
            <Typography
              color={portfolioData.dayChange >= 0 ? 'success.main' : 'error.main'}
              variant="subtitle1"
            >
              {portfolioData.dayChange >= 0 ? '+' : ''}
              {portfolioData.dayChange.toFixed(2)}%
            </Typography>
          </Card>
        </Grid>

        {/* Key Metrics Cards */}
        <Grid item xs={12} md={8}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={4}>
              <Card sx={{ p: 2 }}>
                <Typography variant="subtitle2">Annual Return</Typography>
                <Typography variant="h6">
                  {portfolioData.annualReturn.toFixed(2)}%
                </Typography>
              </Card>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Card sx={{ p: 2 }}>
                <Typography variant="subtitle2">Sharpe Ratio</Typography>
                <Typography variant="h6">
                  {portfolioData.sharpeRatio.toFixed(2)}
                </Typography>
              </Card>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Card sx={{ p: 2 }}>
                <Typography variant="subtitle2">Volatility</Typography>
                <Typography variant="h6">
                  {portfolioData.volatility.toFixed(2)}%
                </Typography>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Chart Controls and Main Chart */}
        <Grid item xs={12}>
          <Card sx={{ p: 2 }}>
            <ChartControls
              timeRange={timeRange}
              selectedIndicators={selectedIndicators}
              benchmark={benchmark}
              onTimeRangeChange={setTimeRange}
              onIndicatorChange={setSelectedIndicators}
              onBenchmarkChange={setBenchmark}
            />
            <Box ref={chartContainerRef} sx={{ width: '100%', height: 400 }} />
          </Card>
        </Grid>

        {/* Portfolio Composition */}
        <Grid item xs={12} md={6}>
          <PortfolioComposition positions={portfolioData.positions} />
        </Grid>

        {/* Risk/Return Plot */}
        <Grid item xs={12} md={6}>
          <RiskReturnPlot positions={portfolioData.positions} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default PortfolioDashboard;
