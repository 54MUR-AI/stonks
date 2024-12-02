from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr
from typing import List
import os
from dotenv import load_dotenv
import asyncio
import yfinance as yf
from datetime import datetime

load_dotenv()

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", "587")),
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

fastmail = FastMail(conf)

async def send_alert_email(email: EmailStr, symbol: str, condition: str, target_price: float, current_price: float):
    """Send email alert when price condition is met"""
    direction = "above" if condition == "above" else "below"
    
    html = f"""
    <h2>Stonks Price Alert</h2>
    <p>Your price alert for {symbol} has been triggered!</p>
    <p>The price is now {direction} your target of ${target_price:.2f}</p>
    <p>Current price: ${current_price:.2f}</p>
    <p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    """
    
    message = MessageSchema(
        subject=f"Stonks Alert: {symbol} Price {direction.capitalize()} ${target_price:.2f}",
        recipients=[email],
        body=html,
        subtype="html"
    )
    
    await fastmail.send_message(message)

async def send_portfolio_summary(email: EmailStr, portfolio_metrics: dict):
    """Send daily portfolio summary email"""
    positions_html = ""
    for pos in portfolio_metrics['positions']:
        positions_html += f"""
        <tr>
            <td>{pos['symbol']}</td>
            <td>${pos['current_price']:.2f}</td>
            <td>${pos['market_value']:.2f}</td>
            <td>{pos['unrealized_pl_pct']:.2f}%</td>
            <td>{pos['daily_return']:.2f}%</td>
        </tr>
        """
    
    html = f"""
    <h2>Daily Portfolio Summary</h2>
    <h3>Portfolio Overview</h3>
    <p>Total Value: ${portfolio_metrics['total_value']:.2f}</p>
    <p>Total P&L: ${portfolio_metrics['unrealized_pl']:.2f} ({portfolio_metrics['unrealized_pl_pct']:.2f}%)</p>
    <p>Daily Return: {portfolio_metrics['daily_return']:.2f}%</p>
    <p>Portfolio Beta: {portfolio_metrics['beta']:.2f}</p>
    <p>Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}</p>
    
    <h3>Position Details</h3>
    <table border="1">
        <tr>
            <th>Symbol</th>
            <th>Price</th>
            <th>Value</th>
            <th>P&L %</th>
            <th>Daily Return</th>
        </tr>
        {positions_html}
    </table>
    
    <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    """
    
    message = MessageSchema(
        subject="Stonks Daily Portfolio Summary",
        recipients=[email],
        body=html,
        subtype="html"
    )
    
    await fastmail.send_message(message)

async def check_price_alerts(db_session, alert_service):
    """Background task to check price alerts"""
    while True:
        try:
            active_alerts = alert_service.get_active_alerts(db_session)
            
            for alert in active_alerts:
                ticker = yf.Ticker(alert.symbol)
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                
                if (alert.condition == "above" and current_price > alert.price) or \
                   (alert.condition == "below" and current_price < alert.price):
                    # Send alert email
                    await send_alert_email(
                        alert.owner.email,
                        alert.symbol,
                        alert.condition,
                        alert.price,
                        current_price
                    )
                    
                    # Deactivate alert
                    alert_service.deactivate_alert(db_session, alert.id)
            
        except Exception as e:
            print(f"Error checking price alerts: {str(e)}")
        
        await asyncio.sleep(60)  # Check every minute
