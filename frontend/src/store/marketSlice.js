import { createSlice } from '@reduxjs/toolkit';
import { SMA, RSI, MACD } from 'technicalindicators';

const initialState = {
    symbols: {},  // { symbol: { data: [], indicators: {}, lastUpdate: timestamp } }
    activeSymbol: null,
    errors: {},
};

const calculateIndicators = (data, period = 14) => {
    if (!data || data.length < period) return null;

    const closes = data.map(candle => candle.close);
    
    // Calculate SMA
    const sma = SMA.calculate({
        period,
        values: closes,
    });

    // Calculate RSI
    const rsi = RSI.calculate({
        period,
        values: closes,
    });

    // Calculate MACD
    const macd = MACD.calculate({
        fastPeriod: 12,
        slowPeriod: 26,
        signalPeriod: 9,
        values: closes,
    });

    return {
        sma: sma[sma.length - 1],
        rsi: rsi[rsi.length - 1],
        macd: macd[macd.length - 1],
    };
};

export const marketSlice = createSlice({
    name: 'market',
    initialState,
    reducers: {
        setActiveSymbol: (state, action) => {
            state.activeSymbol = action.payload;
        },
        updateMarketData: (state, action) => {
            const { symbol, data } = action.payload;
            
            if (!state.symbols[symbol]) {
                state.symbols[symbol] = {
                    data: [],
                    indicators: {},
                    lastUpdate: null,
                };
            }

            // Add new data point
            state.symbols[symbol].data.push(data);
            
            // Keep only last 100 data points for memory efficiency
            if (state.symbols[symbol].data.length > 100) {
                state.symbols[symbol].data.shift();
            }

            // Update indicators
            state.symbols[symbol].indicators = calculateIndicators(
                state.symbols[symbol].data
            );
            
            state.symbols[symbol].lastUpdate = Date.now();
        },
        setError: (state, action) => {
            const { symbol, error } = action.payload;
            state.errors[symbol] = error;
        },
        clearError: (state, action) => {
            const symbol = action.payload;
            delete state.errors[symbol];
        },
    },
});

export const {
    setActiveSymbol,
    updateMarketData,
    setError,
    clearError,
} = marketSlice.actions;

// Selectors
export const selectActiveSymbol = (state) => state.market.activeSymbol;
export const selectSymbolData = (state, symbol) => state.market.symbols[symbol]?.data || [];
export const selectSymbolIndicators = (state, symbol) => state.market.symbols[symbol]?.indicators || null;
export const selectSymbolError = (state, symbol) => state.market.errors[symbol];

export default marketSlice.reducer;
