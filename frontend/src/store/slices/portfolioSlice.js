import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

// Async thunks
export const fetchPortfolios = createAsyncThunk(
    'portfolio/fetchPortfolios',
    async (_, { rejectWithValue }) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/portfolios/`);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

export const fetchPortfolioAnalytics = createAsyncThunk(
    'portfolio/fetchAnalytics',
    async (portfolioId, { rejectWithValue }) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/portfolios/${portfolioId}/analytics`);
            return { portfolioId, data: response.data };
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

export const fetchPortfolioCorrelation = createAsyncThunk(
    'portfolio/fetchCorrelation',
    async (portfolioId, { rejectWithValue }) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/portfolios/${portfolioId}/correlation`);
            return { portfolioId, data: response.data };
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

export const rebalancePortfolio = createAsyncThunk(
    'portfolio/rebalance',
    async ({ portfolioId, weights, tolerance }, { rejectWithValue }) => {
        try {
            const response = await axios.post(
                `${API_BASE_URL}/portfolios/${portfolioId}/rebalance`,
                { weights, tolerance }
            );
            return { portfolioId, data: response.data };
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

export const optimizePortfolio = createAsyncThunk(
    'portfolio/optimize',
    async ({ portfolioId, riskTolerance }, { rejectWithValue }) => {
        try {
            const response = await axios.get(
                `${API_BASE_URL}/portfolios/${portfolioId}/optimize`,
                { params: { risk_tolerance: riskTolerance } }
            );
            return { portfolioId, data: response.data };
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

export const sendPortfolioSummary = createAsyncThunk(
    'portfolio/sendSummary',
    async (portfolioId, { rejectWithValue }) => {
        try {
            const response = await axios.post(
                `${API_BASE_URL}/portfolios/${portfolioId}/email-summary`
            );
            return { portfolioId, data: response.data };
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

const portfolioSlice = createSlice({
    name: 'portfolio',
    initialState: {
        portfolios: [],
        analytics: {},
        correlations: {},
        rebalancing: {},
        optimization: {},
        loading: false,
        error: null,
        lastUpdated: null
    },
    reducers: {
        clearError: (state) => {
            state.error = null;
        },
        updatePortfolioLocal: (state, action) => {
            const index = state.portfolios.findIndex(p => p.id === action.payload.id);
            if (index !== -1) {
                state.portfolios[index] = action.payload;
            }
        }
    },
    extraReducers: (builder) => {
        builder
            // Fetch Portfolios
            .addCase(fetchPortfolios.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchPortfolios.fulfilled, (state, action) => {
                state.loading = false;
                state.portfolios = action.payload;
                state.lastUpdated = new Date().toISOString();
            })
            .addCase(fetchPortfolios.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            
            // Fetch Analytics
            .addCase(fetchPortfolioAnalytics.fulfilled, (state, action) => {
                state.analytics[action.payload.portfolioId] = action.payload.data;
            })
            
            // Fetch Correlation
            .addCase(fetchPortfolioCorrelation.fulfilled, (state, action) => {
                state.correlations[action.payload.portfolioId] = action.payload.data;
            })
            
            // Rebalance Portfolio
            .addCase(rebalancePortfolio.pending, (state) => {
                state.loading = true;
            })
            .addCase(rebalancePortfolio.fulfilled, (state, action) => {
                state.loading = false;
                state.rebalancing[action.payload.portfolioId] = action.payload.data;
            })
            .addCase(rebalancePortfolio.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            
            // Optimize Portfolio
            .addCase(optimizePortfolio.fulfilled, (state, action) => {
                state.optimization[action.payload.portfolioId] = action.payload.data;
            })
            
            // Handle generic errors
            .addMatcher(
                (action) => action.type.endsWith('/rejected'),
                (state, action) => {
                    state.loading = false;
                    state.error = action.payload;
                }
            );
    }
});

export const { clearError, updatePortfolioLocal } = portfolioSlice.actions;

// Selectors
export const selectAllPortfolios = (state) => state.portfolio.portfolios;
export const selectPortfolioById = (state, portfolioId) => 
    state.portfolio.portfolios.find(p => p.id === portfolioId);
export const selectPortfolioAnalytics = (state, portfolioId) =>
    state.portfolio.analytics[portfolioId];
export const selectPortfolioCorrelation = (state, portfolioId) =>
    state.portfolio.correlations[portfolioId];
export const selectRebalancingSummary = (state, portfolioId) =>
    state.portfolio.rebalancing[portfolioId];
export const selectOptimizationResult = (state, portfolioId) =>
    state.portfolio.optimization[portfolioId];
export const selectPortfolioLoading = (state) => state.portfolio.loading;
export const selectPortfolioError = (state) => state.portfolio.error;

export default portfolioSlice.reducer;
