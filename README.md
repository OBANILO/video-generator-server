# Video Generator Server

This is a Flask server that replaces the private server at 72.60.47.186.
It generates videos from image + audio using FFmpeg.

## Endpoints

- `POST /generate` — Start video generation
- `GET /status/<api_key>` — Check job status
- `GET /videos/<job_id>/<filename>` — Download generated video
- `POST /process-audio` — Split audio into segments
- `GET /audio_segments/<filename>` — Download audio segment
- `GET /health` — Health check

## Deploy on Render.com (FREE)

### Step 1 — Create GitHub Repository
1. Go to https://github.com and create a FREE account if you don't have one
2. Click "New repository"
3. Name it: `video-generator-server`
4. Make it Public
5. Click "Create repository"
6. Upload all 3 files: app.py, requirements.txt, render.yaml

### Step 2 — Deploy on Render
1. Go to https://render.com and create a FREE account
2. Click "New" → "Web Service"
3. Connect your GitHub account
4. Select your `video-generator-server` repository
5. Render will auto-detect the render.yaml config
6. Click "Deploy"
7. Wait 3-5 minutes for deployment

### Step 3 — Get your server URL
After deployment, Render gives you a URL like:
`https://video-generator-server.onrender.com`

### Step 4 — Update your WordPress plugin
In the plugin PHP file, replace:
`http://72.60.47.186:80/generate`
with:
`https://video-generator-server.onrender.com/generate`

And replace:
`http://72.60.47.186:2095/process-audio`
with:
`https://video-generator-server.onrender.com/process-audio`

And replace:
`http://72.60.47.186:2095/audio_segments/`
with:
`https://video-generator-server.onrender.com/audio_segments/`

And replace:
`http://72.60.47.186:80/status/`
with:
`https://video-generator-server.onrender.com/status/`

## Notes
- Free Render plan may sleep after 15 min of inactivity (first request will be slow)
- FFmpeg is pre-installed on Render's Linux environment
- Videos are stored temporarily in /tmp (cleared on restart)
