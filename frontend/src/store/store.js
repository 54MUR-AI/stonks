import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import marketReducer from './marketSlice';

const persistConfig = {
    key: 'root',
    storage,
    whitelist: ['symbols'], // Only persist market data
};

const persistedReducer = persistReducer(persistConfig, marketReducer);

export const store = configureStore({
    reducer: {
        market: persistedReducer,
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            serializableCheck: false, // Disable for redux-persist
        }),
});

export const persistor = persistStore(store);
