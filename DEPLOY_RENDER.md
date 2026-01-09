# Deploying Smart Seva API to Render.io

This guide will walk you through deploying your FastAPI server to Render.com.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com) (free tier available)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **API Keys Ready**:
   - Groq API Key (for LLM) - [Get it here](https://console.groq.com)
   - Google Cloud Vision API Key (optional, for OCR fallback) - [Get it here](https://cloud.google.com/vision/docs/setup)

## Step-by-Step Deployment Guide

### Step 1: Prepare Your Repository

1. **Commit all changes** to your repository:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Verify the following files exist**:
   - `render.yaml` (deployment configuration)
   - `pyproject.toml` (Python dependencies)
   - `src/api/main.py` (FastAPI app entry point)

### Step 2: Create a New Web Service on Render

1. **Log in to Render Dashboard**: Go to [dashboard.render.com](https://dashboard.render.com)

2. **Create New Web Service**:
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub repository (authorize Render if needed)
   - Select the repository containing your server code

3. **Configure the Service**:
   - **Name**: `smart-seva-api` (or your preferred name)
   - **Region**: Choose closest to your users (e.g., Oregon, Frankfurt)
   - **Branch**: `main` (or your deployment branch)
   - **Root Directory**: `server` (since your server code is in the `server/` folder)
   - **Environment**: `Python 3`
   - **Build Command**: 
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.cargo/bin:$PATH" && uv sync
     ```
   - **Start Command**:
     ```bash
     uv run uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan**: Choose based on your needs:
     - **Free**: Limited resources, spins down after inactivity
     - **Starter ($7/month)**: Always on, better performance
     - **Standard ($25/month)**: More resources, better for production

### Step 3: Configure Environment Variables

In the Render dashboard, go to your service → **Environment** tab and add:

#### Required Environment Variables:

```bash
# LLM Provider (use Groq for production)
LLM_PROVIDER=groq

# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile

# OCR Configuration
OCR_PROVIDER=easyocr
OCR_LANGUAGES=["en", "te"]
OCR_GPU=false  # Render doesn't provide GPU on free/starter plans
OCR_LIGHTWEIGHT=true
OCR_MAX_IMAGE_DIMENSION=2000
OCR_MIN_CONFIDENCE=0.5
OCR_BI_LANG=false

# Google Vision API (optional, for OCR fallback)
# You can either:
# Option 1: Set as JSON string
GOOGLE_VISION_API_KEY={"type": "service_account", "project_id": "...", ...}

# Option 2: Set path to JSON file (if you upload it)
# GOOGLE_VISION_API_KEY=/opt/render/project/src/aprtgs-bee0df2287cf.json

# Logging
LOG_LEVEL=INFO
```

#### Important Notes:

- **LLM Provider**: Use `groq` (not `ollama`) since Ollama requires a local server
- **GPU**: Set `OCR_GPU=false` as Render free/starter plans don't have GPU access
- **Google Vision API Key**: 
  - You can paste the entire JSON content as a string
  - Or upload the JSON file and reference its path
  - Make sure to escape quotes properly if using JSON string

### Step 4: Handle Google Cloud Vision API Key File

Since Render doesn't support file uploads directly, you have two options:

#### Option A: Use Environment Variable (Recommended)

1. Read your `aprtgs-bee0df2287cf.json` file
2. Copy its entire content
3. In Render dashboard, add environment variable:
   ```
   GOOGLE_VISION_API_KEY={"type": "service_account", "project_id": "...", ...}
   ```
   (Paste the entire JSON content)

#### Option B: Embed in Repository (Less Secure)

1. Keep the file in your repository (already there)
2. Update your code to read from the file path
3. Set environment variable:
   ```
   GOOGLE_VISION_API_KEY=/opt/render/project/src/aprtgs-bee0df2287cf.json
   ```

**⚠️ Security Note**: Option B is less secure. Consider using Option A or Render's secret management.

### Step 5: Deploy

1. **Save Configuration**: Click "Save Changes" in Render dashboard
2. **Manual Deploy**: Click "Manual Deploy" → "Deploy latest commit"
3. **Monitor Logs**: Watch the build and deployment logs in real-time

### Step 6: Verify Deployment

Once deployed, your API will be available at:
```
https://smart-seva-api.onrender.com
```

Test the endpoints:

```bash
# Health check
curl https://smart-seva-api.onrender.com/health

# Root endpoint
curl https://smart-seva-api.onrender.com/

# API docs
# Visit: https://smart-seva-api.onrender.com/docs
```

## Using render.yaml (Alternative Method)

If you prefer using the `render.yaml` file:

1. **Ensure render.yaml is in your repository root** (or adjust path in Render)
2. **In Render Dashboard**:
   - Go to "New +" → "Blueprint"
   - Connect your repository
   - Render will automatically detect and use `render.yaml`
3. **Set sensitive environment variables** in the dashboard (those marked `sync: false`)

## Troubleshooting

### Build Fails

**Issue**: Build command fails
- **Solution**: Check that `uv` installation command works. You may need to adjust the build command.

**Issue**: Python version mismatch
- **Solution**: Ensure `PYTHON_VERSION=3.13.0` is set in environment variables.

### Runtime Errors

**Issue**: Port binding error
- **Solution**: Ensure start command uses `$PORT` environment variable (Render provides this automatically).

**Issue**: Module not found errors
- **Solution**: Verify `uv sync` completed successfully. Check build logs.

**Issue**: OCR models not downloading
- **Solution**: First request may take longer as models download. Check logs for progress.

### Performance Issues

**Issue**: Slow OCR processing
- **Solution**: 
  - Upgrade to a higher plan (Standard or Pro)
  - Use `OCR_LIGHTWEIGHT=true`
  - Consider using Google Vision API as primary OCR provider

**Issue**: Service spins down (Free plan)
- **Solution**: 
  - Upgrade to Starter plan ($7/month) for always-on service
  - Or use a service like UptimeRobot to ping your service periodically

### Environment Variable Issues

**Issue**: Environment variables not being read
- **Solution**: 
  - Ensure variables are set in Render dashboard (not just in render.yaml)
  - Restart the service after adding variables
  - Check variable names match exactly (case-sensitive)

## Production Recommendations

1. **Use Starter Plan or Higher**: Free plan spins down after inactivity
2. **Set Up Monitoring**: Use Render's built-in metrics or integrate external monitoring
3. **Configure CORS Properly**: Update CORS origins in `src/api/main.py` to your frontend domain
4. **Use Environment-Specific Configs**: Consider separate services for staging/production
5. **Set Up Health Checks**: Render automatically monitors `/health` endpoint
6. **Enable Auto-Deploy**: Configure automatic deployments from your main branch
7. **Set Up Log Aggregation**: Use Render's log streaming or integrate external services

## Updating Your Deployment

To update your deployed service:

1. **Push changes to GitHub**:
   ```bash
   git add .
   git commit -m "Update API"
   git push origin main
   ```

2. **Render will auto-deploy** (if enabled) or manually trigger deployment in dashboard

## Cost Estimation

- **Free Plan**: $0/month (limited resources, spins down)
- **Starter Plan**: $7/month (always on, 512MB RAM)
- **Standard Plan**: $25/month (2GB RAM, better performance)

Choose based on your expected traffic and requirements.

## Next Steps

After deployment:

1. **Update Frontend**: Point your web app to the Render URL
2. **Test All Endpoints**: Verify all API endpoints work correctly
3. **Monitor Performance**: Check Render dashboard for metrics
4. **Set Up Custom Domain** (optional): Configure custom domain in Render settings

## Support

- **Render Documentation**: [render.com/docs](https://render.com/docs)
- **Render Community**: [community.render.com](https://community.render.com)
- **FastAPI Docs**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

---

**Note**: Remember to keep your API keys secure. Never commit them to your repository. Always use Render's environment variables for sensitive data.
