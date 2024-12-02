import React from 'react';
import { useSelector } from 'react-redux';
import { selectSymbolIndicators } from '../store/marketSlice';
import { Box, Paper, Typography, Grid } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

const IndicatorCard = ({ title, value, interpretation }) => {
    const getColor = () => {
        if (interpretation === 'bullish') return 'success.main';
        if (interpretation === 'bearish') return 'error.main';
        return 'info.main';
    };

    const Icon = interpretation === 'bullish' ? TrendingUpIcon : TrendingDownIcon;

    return (
        <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
            <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="h6" color="text.secondary">
                    {title}
                </Typography>
                <Icon sx={{ color: getColor() }} />
            </Box>
            <Typography variant="h4" component="div" sx={{ mt: 1 }}>
                {typeof value === 'number' ? value.toFixed(2) : 'N/A'}
            </Typography>
        </Paper>
    );
};

const TechnicalIndicators = ({ symbol }) => {
    const indicators = useSelector((state) => selectSymbolIndicators(state, symbol));

    if (!indicators) {
        return (
            <Paper elevation={3} sx={{ p: 2 }}>
                <Typography variant="body1" color="text.secondary">
                    Calculating indicators...
                </Typography>
            </Paper>
        );
    }

    const interpretRSI = (value) => {
        if (value > 70) return 'bearish';
        if (value < 30) return 'bullish';
        return 'neutral';
    };

    const interpretMACD = (macd) => {
        if (!macd) return 'neutral';
        return macd.histogram > 0 ? 'bullish' : 'bearish';
    };

    return (
        <Box sx={{ mt: 2 }}>
            <Typography variant="h5" gutterBottom>
                Technical Indicators
            </Typography>
            <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                    <IndicatorCard
                        title="SMA"
                        value={indicators.sma}
                        interpretation={indicators.sma > 0 ? 'bullish' : 'bearish'}
                    />
                </Grid>
                <Grid item xs={12} sm={4}>
                    <IndicatorCard
                        title="RSI"
                        value={indicators.rsi}
                        interpretation={interpretRSI(indicators.rsi)}
                    />
                </Grid>
                <Grid item xs={12} sm={4}>
                    <IndicatorCard
                        title="MACD"
                        value={indicators.macd?.histogram}
                        interpretation={interpretMACD(indicators.macd)}
                    />
                </Grid>
            </Grid>
        </Box>
    );
};

export default TechnicalIndicators;
