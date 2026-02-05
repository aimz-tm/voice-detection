# Deployment Guide for Voice Detection API

## **Option 1: Vercel (Quick, but Limited)**

### Pros:
- Free tier available
- Easy to deploy
- Automatic HTTPS
- Simple git integration

### Cons:
- **Timeout limit**: 10 seconds (Pro) - audio processing may exceed this
- **Cold starts**: First request can be slow (5-10 sec)
- **Memory limit**: 512MB
- Large dependencies (librosa) slow down deployment

### Steps:

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel
   ```

4. **For production**:
   ```bash
   vercel --prod
   ```

---

## **Option 2: Railway.app (RECOMMENDED for this project)**

### Pros:
- **Better for audio processing** (long-running tasks)
- Generous free tier ($5/month credit)
- Supports persistent runtime
- Easy database integration
- Better cold start handling

### Steps:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Go to** https://railway.app

3. **Create Project** → Click "Deploy from GitHub"

4. **Select your repo**

5. **Railway auto-detects Python** → Deploy

6. **Environment Variables**: Add in Railway dashboard if needed

---

## **Option 3: Render (Great Alternative)**

### Pros:
- Free tier with 15GB monthly bandwidth
- Great UI
- Persistent storage
- No cold sleep
- Good for FastAPI apps

### Steps:

1. **Push to GitHub** (as above)

2. **Go to** https://render.com

3. **New → Web Service**

4. **Connect GitHub repo**

5. **Settings**:
   - **Runtime**: Python 3.11
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 8000`

6. **Deploy**

---

## **Option 4: AWS Lambda (Scalable)**

### Pros:
- Free tier: 1 million requests/month
- Scales automatically
- Production-ready

### Cons:
- More complex setup
- Requires AWS account

### Minimal Setup:
1. Use AWS SAM or Zappa for FastAPI
2. Package with dependencies
3. Deploy to Lambda

```bash
pip install zappa
zappa init
zappa deploy dev
```

---

## **Option 5: Docker + Heroku (No longer free, but alternative)**

Use Railway or Render with Docker:

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Then deploy to any container service (Railway, Render, AWS ECS, etc.)

---

## **Recommendation for Your Project**

**Use Railway.app or Render.com** because:
- ✅ Audio processing requires 30-60 sec timeout (not 10 sec)
- ✅ No cold start issues
- ✅ Easy deployment from GitHub
- ✅ Free tier is sufficient
- ✅ Better for ML models like yours

---

## **Local Testing Before Deploy**

```bash
# Test locally
uvicorn main:app --host 0.0.0.0 --port 8000

# Then visit: http://localhost:8000/docs
```

---

## **Important Notes**

1. **Model File**: Upload `voice_detector_model.pkl` to GitHub with `.gitignore` exception, or regenerate on startup
2. **FFmpeg**: Some services require system packages - specify in build config
3. **Environment Variables**: Store API keys in deployment platform's secret manager
4. **Size Limits**: Audio file uploads may be limited - add validation in your API

---

## **Comparison Table**

| Platform | Free Tier | Timeout | Cold Start | Best For |
|----------|-----------|---------|-----------|----------|
| Vercel | ✅ | 10s | Slow | Frontend |
| Railway | ✅ | 30m | Fast | Python APIs |
| Render | ✅ | 30m | Fast | FastAPI |
| AWS Lambda | ✅ (1M req) | 15m (max) | Very Slow | Scalability |
| Heroku | ❌ (paid) | 30s | Normal | Legacy |

---

## **Quick Deploy Commands**

**Railway**:
```bash
# Install railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway up
```

**Render** (via GitHub):
- Connect GitHub repo
- Select repo
- Auto-configures for Python

---

**I recommend Railway or Render for your voice detection API.** They handle long-running audio processing tasks better than Vercel.
