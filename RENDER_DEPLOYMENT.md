# STONKS Render.com Deployment Guide

## 🚀 Quick Deploy

This repository is configured for automatic deployment to Render.com using the `render.yaml` blueprint.

## 📋 Prerequisites

1. Render.com account connected to GitHub (54MUR-AI organization)
2. Repository pushed to GitHub

## 🔧 Deployment Steps

### Option 1: Blueprint Deploy (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub account if not already connected
4. Select the **`54MUR-AI/stonks`** repository
5. Click **"Apply"**
6. Render will automatically create:
   - `stonks-api` - Backend FastAPI service
   - `stonks-app` - Frontend React app
   - `stonks-db` - PostgreSQL database

### Option 2: Manual Deploy

#### Backend API:
1. New → Web Service
2. Connect `54MUR-AI/stonks` repo
3. Settings:
   - **Name:** `stonks-api`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free

#### Frontend:
1. New → Static Site
2. Connect `54MUR-AI/stonks` repo
3. Settings:
   - **Name:** `stonks-app`
   - **Build Command:** `cd frontend && npm install && npm run build`
   - **Publish Directory:** `frontend/dist`
   - **Plan:** Free

#### Database:
1. New → PostgreSQL
2. Settings:
   - **Name:** `stonks-db`
   - **Plan:** Free

## 🔐 Environment Variables

### Backend (`stonks-api`):

**Auto-generated:**
- `DATABASE_URL` - From PostgreSQL database
- `SECRET_KEY` - Auto-generated secure key

**Required (add manually):**
```
ENVIRONMENT=production
ALLOWED_ORIGINS=https://stonks-app.onrender.com,https://54mur-ai.github.io
```

**Optional (for full features):**
```
# News Intelligence Service
SUMMARIZER_PROVIDER=openai
OPENAI_API_KEY=your_openai_key

# Or use Anthropic
SUMMARIZER_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Or use Ollama (local)
SUMMARIZER_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434

# Email notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### Frontend (`stonks-app`):

```
VITE_API_URL=https://stonks-api.onrender.com
```

## 📊 Database Setup

After deployment, run migrations:

```bash
# Connect to your backend shell on Render
python -c "from backend.database import Base, engine; Base.metadata.create_all(bind=engine)"
```

Or use the Render Shell feature in the dashboard.

## 🌐 Access Your App

- **Frontend:** `https://stonks-app.onrender.com`
- **Backend API:** `https://stonks-api.onrender.com`
- **API Docs:** `https://stonks-api.onrender.com/docs`

## ⚠️ Free Tier Limitations

- **Spin down after 15 minutes** of inactivity
- **Cold start:** ~30 seconds on first request
- **Database:** 90 days free, then $7/month
- **750 hours/month** of runtime

## 🔄 Auto-Deploy

Every push to `main` branch triggers automatic redeployment.

## 🐛 Troubleshooting

### Build Fails
- Check Python version (3.13.0)
- Verify all dependencies in `requirements.txt`
- Check build logs in Render dashboard

### Database Connection Issues
- Ensure `DATABASE_URL` is set
- Check database is running
- Verify connection string format

### CORS Errors
- Update `ALLOWED_ORIGINS` environment variable
- Include both Render URL and GitHub Pages URL

## 📝 Notes

- First deployment takes ~5-10 minutes
- Subsequent deploys are faster (~2-3 minutes)
- Free tier services spin down after inactivity
- Keep services active with uptime monitors (e.g., UptimeRobot)

## 🔗 Useful Links

- [Render Documentation](https://render.com/docs)
- [FastAPI Deployment](https://render.com/docs/deploy-fastapi)
- [React Deployment](https://render.com/docs/deploy-create-react-app)
