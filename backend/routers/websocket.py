from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..websockets.market_data import get_market_data_manager, MarketDataManager
from ..auth import get_current_user

router = APIRouter()

@router.websocket("/ws/portfolio/{portfolio_id}")
async def portfolio_websocket(
    websocket: WebSocket,
    portfolio_id: int,
    db: Session = Depends(get_db),
    market_data_manager: MarketDataManager = Depends(get_market_data_manager)
):
    try:
        await market_data_manager.connect(websocket, portfolio_id)
        await market_data_manager.subscribe_portfolio(portfolio_id, db)
        
        while True:
            # Keep the connection alive and handle any client messages
            data = await websocket.receive_text()
            try:
                # Handle any client-side messages here
                pass
            except Exception as e:
                print(f"Error handling websocket message: {e}")
                
    except WebSocketDisconnect:
        market_data_manager.disconnect(websocket, portfolio_id)
    except Exception as e:
        print(f"Error in websocket connection: {e}")
        market_data_manager.disconnect(websocket, portfolio_id)
