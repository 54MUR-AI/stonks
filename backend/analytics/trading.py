import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time, timedelta
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from ..schemas.trading import (
    TradeOrder, ExecutionParams, OrderStatus, OrderUpdate,
    ExecutionResult, MarketCondition, ExecutionStrategy
)
from .market_data import get_market_summary

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self):
        self.market_hours = {
            "NYSE": {
                "open": time(9, 30),
                "close": time(16, 0)
            }
        }
        
    async def execute_trades(
        self,
        orders: List[TradeOrder],
        execution_params: ExecutionParams,
        dry_run: bool = True
    ) -> ExecutionResult:
        """Execute a list of trades according to the specified strategy"""
        try:
            # Validate market hours
            if not self._validate_market_hours("NYSE"):
                return ExecutionResult(
                    success=False,
                    orders=[],
                    total_cost=0,
                    average_price={},
                    execution_time=0,
                    errors=["Market is closed"]
                )
            
            # Get current market conditions
            symbols = [order.symbol for order in orders]
            market_data = get_market_summary(symbols)
            
            if execution_params.strategy == ExecutionStrategy.IMMEDIATE:
                return await self._execute_immediate(orders, market_data, dry_run)
            elif execution_params.strategy == ExecutionStrategy.TWAP:
                return await self._execute_twap(orders, execution_params, market_data, dry_run)
            elif execution_params.strategy == ExecutionStrategy.VWAP:
                return await self._execute_vwap(orders, execution_params, market_data, dry_run)
            else:  # SMART
                return await self._execute_smart(orders, execution_params, market_data, dry_run)
                
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            return ExecutionResult(
                success=False,
                orders=[],
                total_cost=0,
                average_price={},
                execution_time=0,
                errors=[str(e)]
            )
    
    def _validate_market_hours(self, market: str) -> bool:
        """Check if market is open"""
        if market not in self.market_hours:
            return False
            
        now = datetime.now().time()
        hours = self.market_hours[market]
        return hours["open"] <= now <= hours["close"]
    
    async def _execute_immediate(
        self,
        orders: List[TradeOrder],
        market_data: Dict[str, Dict],
        dry_run: bool
    ) -> ExecutionResult:
        """Execute all orders immediately at market price"""
        start_time = datetime.now()
        executed_orders = []
        total_cost = 0
        average_prices = {}
        warnings = []
        
        for order in orders:
            symbol_data = market_data.get(order.symbol, {})
            if not symbol_data:
                warnings.append(f"No market data available for {order.symbol}")
                continue
                
            # Simulate execution
            execution_price = symbol_data.get("price", 0)
            if execution_price == 0:
                warnings.append(f"Invalid price for {order.symbol}")
                continue
                
            cost = execution_price * order.quantity
            total_cost += cost
            average_prices[order.symbol] = execution_price
            
            executed_orders.append({
                "order_id": f"order_{len(executed_orders)}",
                "status": OrderStatus.FILLED if not dry_run else OrderStatus.PENDING,
                "filled_quantity": order.quantity,
                "average_price": execution_price,
                "last_price": execution_price,
                "last_quantity": order.quantity,
                "remaining_quantity": 0,
                "timestamp": datetime.now(),
                "message": "Simulated execution" if dry_run else "Order filled"
            })
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ExecutionResult(
            success=True,
            orders=executed_orders,
            total_cost=total_cost,
            average_price=average_prices,
            execution_time=execution_time,
            warnings=warnings
        )
    
    async def _execute_twap(
        self,
        orders: List[TradeOrder],
        params: ExecutionParams,
        market_data: Dict[str, Dict],
        dry_run: bool
    ) -> ExecutionResult:
        """Execute orders using Time-Weighted Average Price strategy"""
        start_time = datetime.now()
        executed_orders = []
        total_cost = 0
        average_prices = {}
        warnings = []
        
        # Calculate time windows
        start = params.start_time or datetime.now().time()
        end = params.end_time or self.market_hours["NYSE"]["close"]
        duration = datetime.combine(datetime.today(), end) - datetime.combine(datetime.today(), start)
        num_intervals = max(1, int(duration.total_seconds() / 300))  # 5-minute intervals
        
        for order in orders:
            symbol_data = market_data.get(order.symbol, {})
            if not symbol_data:
                warnings.append(f"No market data available for {order.symbol}")
                continue
            
            # Calculate size per interval
            size_per_interval = max(
                params.min_trade_size or 1,
                order.quantity // num_intervals
            )
            
            # Simulate TWAP execution
            remaining = order.quantity
            total_value = 0
            
            for _ in range(num_intervals):
                if remaining <= 0:
                    break
                    
                interval_size = min(size_per_interval, remaining)
                execution_price = symbol_data.get("price", 0)
                
                if execution_price == 0:
                    warnings.append(f"Invalid price for {order.symbol}")
                    continue
                
                interval_cost = execution_price * interval_size
                total_value += interval_cost
                total_cost += interval_cost
                remaining -= interval_size
                
                if not dry_run:
                    # Here we would submit actual trades
                    pass
            
            average_price = total_value / order.quantity
            average_prices[order.symbol] = average_price
            
            executed_orders.append({
                "order_id": f"order_{len(executed_orders)}",
                "status": OrderStatus.FILLED if not dry_run else OrderStatus.PENDING,
                "filled_quantity": order.quantity - remaining,
                "average_price": average_price,
                "last_price": execution_price,
                "last_quantity": interval_size,
                "remaining_quantity": remaining,
                "timestamp": datetime.now(),
                "message": "TWAP simulation complete" if dry_run else "TWAP execution complete"
            })
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ExecutionResult(
            success=True,
            orders=executed_orders,
            total_cost=total_cost,
            average_price=average_prices,
            execution_time=execution_time,
            warnings=warnings
        )
    
    async def _execute_vwap(
        self,
        orders: List[TradeOrder],
        params: ExecutionParams,
        market_data: Dict[str, Dict],
        dry_run: bool
    ) -> ExecutionResult:
        """Execute orders using Volume-Weighted Average Price strategy"""
        start_time = datetime.now()
        executed_orders = []
        total_cost = 0
        average_prices = {}
        warnings = []
        
        for order in orders:
            symbol_data = market_data.get(order.symbol, {})
            if not symbol_data:
                warnings.append(f"No market data available for {order.symbol}")
                continue
            
            volume = symbol_data.get("volume", 0)
            if volume == 0:
                warnings.append(f"No volume data for {order.symbol}")
                continue
            
            # Calculate participation size based on volume
            max_participation = params.max_participation_rate or 0.1  # Default to 10%
            participation_size = int(volume * max_participation)
            execution_size = min(participation_size, order.quantity)
            
            execution_price = symbol_data.get("vwap") or symbol_data.get("price", 0)
            if execution_price == 0:
                warnings.append(f"Invalid price for {order.symbol}")
                continue
            
            cost = execution_price * execution_size
            total_cost += cost
            average_prices[order.symbol] = execution_price
            
            executed_orders.append({
                "order_id": f"order_{len(executed_orders)}",
                "status": OrderStatus.FILLED if not dry_run else OrderStatus.PENDING,
                "filled_quantity": execution_size,
                "average_price": execution_price,
                "last_price": execution_price,
                "last_quantity": execution_size,
                "remaining_quantity": order.quantity - execution_size,
                "timestamp": datetime.now(),
                "message": "VWAP simulation complete" if dry_run else "VWAP execution complete"
            })
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ExecutionResult(
            success=True,
            orders=executed_orders,
            total_cost=total_cost,
            average_price=average_prices,
            execution_time=execution_time,
            warnings=warnings
        )
    
    async def _execute_smart(
        self,
        orders: List[TradeOrder],
        params: ExecutionParams,
        market_data: Dict[str, Dict],
        dry_run: bool
    ) -> ExecutionResult:
        """Execute orders using smart order routing strategy"""
        # Analyze market conditions
        market_conditions = self._analyze_market_conditions(orders, market_data)
        
        # Choose best execution strategy based on conditions
        strategies = []
        for order in orders:
            condition = market_conditions.get(order.symbol, {})
            
            if condition.get("volatility", 0) > 0.02:  # High volatility
                strategies.append(("TWAP", order))
            elif condition.get("spread", 0) > 0.001:  # Wide spread
                strategies.append(("VWAP", order))
            else:
                strategies.append(("IMMEDIATE", order))
        
        # Execute using selected strategies
        results = []
        for strategy, order in strategies:
            if strategy == "TWAP":
                result = await self._execute_twap([order], params, market_data, dry_run)
            elif strategy == "VWAP":
                result = await self._execute_vwap([order], params, market_data, dry_run)
            else:
                result = await self._execute_immediate([order], market_data, dry_run)
            results.append(result)
        
        # Combine results
        total_cost = sum(r.total_cost for r in results)
        all_orders = [order for r in results for order in r.orders]
        all_warnings = [w for r in results for w in r.warnings]
        average_prices = {}
        for r in results:
            average_prices.update(r.average_price)
        
        return ExecutionResult(
            success=all(r.success for r in results),
            orders=all_orders,
            total_cost=total_cost,
            average_price=average_prices,
            execution_time=max(r.execution_time for r in results),
            warnings=all_warnings
        )
    
    def _analyze_market_conditions(
        self,
        orders: List[TradeOrder],
        market_data: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Analyze market conditions for smart order routing"""
        conditions = {}
        for order in orders:
            symbol_data = market_data.get(order.symbol, {})
            if not symbol_data:
                continue
            
            price = symbol_data.get("price", 0)
            bid = symbol_data.get("bid_price", price)
            ask = symbol_data.get("ask_price", price)
            
            conditions[order.symbol] = {
                "volatility": symbol_data.get("volatility", 0),
                "spread": (ask - bid) / price if price > 0 else 0,
                "volume": symbol_data.get("volume", 0)
            }
        
        return conditions
