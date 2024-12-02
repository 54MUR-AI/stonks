import { useEffect, useRef, useCallback } from 'react';
import { useDispatch } from 'react-redux';
import { updateMarketData, setError, clearError } from '../store/marketSlice';

const WEBSOCKET_URL = 'ws://localhost:8000/ws/market';

export const useMarketDataSocket = () => {
    const ws = useRef(null);
    const reconnectTimeout = useRef(null);
    const dispatch = useDispatch();

    const connect = useCallback(() => {
        if (ws.current?.readyState === WebSocket.OPEN) return;

        ws.current = new WebSocket(WEBSOCKET_URL);

        ws.current.onopen = () => {
            console.log('WebSocket connected');
            if (reconnectTimeout.current) {
                clearTimeout(reconnectTimeout.current);
                reconnectTimeout.current = null;
            }
        };

        ws.current.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'market_update') {
                dispatch(updateMarketData({
                    symbol: message.symbol,
                    data: message.data
                }));
                dispatch(clearError(message.symbol));
            } else if (message.type === 'error') {
                dispatch(setError({
                    symbol: message.symbol,
                    error: message.error
                }));
            }
        };

        ws.current.onclose = () => {
            console.log('WebSocket disconnected. Reconnecting...');
            reconnectTimeout.current = setTimeout(connect, 5000);
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            ws.current?.close();
        };
    }, [dispatch]);

    const subscribe = useCallback((symbol) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({
                action: 'subscribe',
                symbol: symbol
            }));
        }
    }, []);

    const unsubscribe = useCallback((symbol) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({
                action: 'unsubscribe',
                symbol: symbol
            }));
        }
    }, []);

    useEffect(() => {
        connect();
        return () => {
            if (ws.current) {
                ws.current.close();
            }
            if (reconnectTimeout.current) {
                clearTimeout(reconnectTimeout.current);
            }
        };
    }, [connect]);

    return { subscribe, unsubscribe };
};
