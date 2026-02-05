# Deploy to Railway.app

## Quick Deploy (3 minutes)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: AI Voice Detection API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/voice-detection.git
git push -u origin main
```

### Step 2: Deploy on Railway
1. Go to https://railway.app/dashboard
2. Click **Create Project**
3. Select **Deploy from GitHub repo**
4. Connect your GitHub account
5. Select **voice-detection** repo
6. Click **Deploy**

Railway will:
- ✅ Auto-detect `Dockerfile`
- ✅ Install FFmpeg (specified in Dockerfile)
- ✅ Install Python dependencies
- ✅ Start the API automatically

### Step 3: Test Your API
Once deployed, Railway shows your public URL:
```
https://your-app-name.railway.app
```

Visit the interactive docs:
```
https://your-app-name.railway.app/docs
```

Or health check:
```bash
curl https://your-app-name.railway.app/
```

---

## Deploy from Command Line (Alternative)

If you prefer CLI:

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Create project
railway init

# 4. Deploy
railway up

# 5. View logs
railway logs
```

---

## Accessing Your API

### REST API
```bash
# Upload audio file
curl -X POST "https://your-app-domain.railway.app/detect-voice" \
  -F "file=@path/to/audio.mp3"
```

### Response Example
```json
{
  "classification": "Human-generated",
  "confidence": 0.9850,
  "explanation": "Natural prosody and spectral variation suggest human speech.",
  "detected_language": "English"
}
```

### Interactive Docs
Open in browser:
```
https://your-app-domain.railway.app/docs
```

---

## Environment Variables (if needed)

In Railway Dashboard:

1. Go to **Variables**
2. Add any secrets (API keys, etc.)
3. They're automatically available in your app

---

## Monitoring & Logs

In Railway Dashboard:

- **Logs**: See real-time application logs
- **Metrics**: CPU, memory, network usage
- **Deployments**: View all deployment history
- **Rollback**: Revert to previous version if needed

---

## Custom Domain (Optional)

1. Go to **Settings** in Railway
2. Add custom domain
3. Update DNS records
4. HTTPS auto-enabled

---

## Troubleshooting

### App crashes after deploy
- Check logs: `railway logs`
- Verify all dependencies in `requirements.txt`
- Ensure port is 8000

### Model file not loading
Railway downloads full git repo including `voice_detector_model.pkl`

### Timeout on large audio
Railway supports up to 30 minutes for requests (plenty for audio processing)

---

## Cost Estimate

Railway Free Tier:
- **$5/month** credit (always free to try)
- Your app fits easily in free tier
- Paid plans: $5-25/month for typical usage

---

## Git Workflow for Updates

After deployment, updates are automatic:

```bash
# Make changes locally
git add .
git commit -m "Update model accuracy"
git push origin main

# Railway auto-deploys! Check logs:
railway logs
```

---

## Support

- Issues? Check Railway docs: https://docs.railway.app
- Deployment status: https://railway.app/dashboard
- Community: https://discord.gg/railway

---

You're all set! Your Voice Detection API will be live in ~2-3 minutes. 🚀
