import pytest
from models import User, Portfolio, PortfolioShare, UserFollow, Comment

@pytest.fixture
def test_user2(db):
    user = User(
        email="test2@example.com",
        username="testuser2",
        hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LHr9v.X9BkAAKJHZe"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@pytest.fixture
def test_user2_token(test_user2):
    return create_access_token(data={"sub": test_user2.email})

@pytest.fixture
def authorized_client2(client, test_user2_token):
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {test_user2_token}"
    }
    return client

@pytest.fixture
def test_portfolio(db, test_user):
    portfolio = Portfolio(
        name="Test Portfolio",
        user_id=test_user.id,
        description="A test portfolio",
        is_public=True
    )
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio

@pytest.fixture
def shared_portfolio(db, test_user, test_user2):
    portfolio = Portfolio(
        name="Shared Portfolio",
        owner_id=test_user.id,
        is_public=True
    )
    db.add(portfolio)
    db.commit()
    
    share = PortfolioShare(
        portfolio_id=portfolio.id,
        shared_with_id=test_user2.id,
        permission="read"
    )
    db.add(share)
    db.commit()
    
    db.refresh(portfolio)
    return portfolio

def test_share_portfolio(authorized_client, test_portfolio, test_user2):
    response = authorized_client.post(
        f"/portfolios/{test_portfolio.id}/share",
        json={
            "user_id": test_user2.id,
            "permission": "read"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["shared_with_id"] == test_user2.id
    assert data["permission"] == "read"

def test_get_shared_portfolios(authorized_client, shared_portfolio):
    response = authorized_client.get("/portfolios/shared")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == shared_portfolio.id

def test_follow_user(authorized_client, test_user2):
    response = authorized_client.post(
        f"/users/{test_user2.id}/follow"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["following_id"] == test_user2.id

def test_get_followers(authorized_client, test_user, test_user2, db):
    # Create follow relationship
    follow = UserFollow(
        follower_id=test_user2.id,
        following_id=test_user.id
    )
    db.add(follow)
    db.commit()
    
    response = authorized_client.get(f"/users/{test_user.id}/followers")
    assert response.status_code == 200
    followers = response.json()
    assert len(followers) == 1
    assert followers[0]["id"] == test_user2.id

def test_comment_on_portfolio(authorized_client, test_portfolio):
    response = authorized_client.post(
        f"/portfolios/{test_portfolio.id}/comments",
        json={"content": "Great portfolio!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Great portfolio!"

def test_get_portfolio_comments(authorized_client, test_portfolio):
    # Create a comment
    comment = Comment(
        portfolio_id=test_portfolio.id,
        user_id=authorized_client.user_id,
        content="Test comment"
    )
    authorized_client.db.add(comment)
    authorized_client.db.commit()
    
    response = authorized_client.get(f"/portfolios/{test_portfolio.id}/comments")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "Test comment"
