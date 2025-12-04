# Deploying AI Peer Review

Multiple deployment options from easiest to most customizable.

---

## 🚀 Option 1: Railway (Easiest - One Click)

Railway offers the simplest deployment with automatic HTTPS.

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

**Steps:**
1. Click the button above or go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway auto-detects the Dockerfile
4. Add your API keys in the Railway dashboard under "Variables"
5. Deploy! You get a free `.up.railway.app` domain with HTTPS

**Cost:** Free tier includes 500 hours/month, ~$5/month for always-on

---

## 🚀 Option 2: Render (Easy - Free Tier)

Render offers a generous free tier and easy setup.

**Steps:**
1. Go to [render.com](https://render.com) and create account
2. New → Web Service → Connect your repo
3. Settings:
   - **Environment:** Docker
   - **Plan:** Free (spins down after 15 min inactivity) or Starter ($7/mo)
4. Add environment variables for API keys
5. Deploy!

**Cost:** Free tier available (with cold starts), $7/month for always-on

---

## 🚀 Option 3: Fly.io (Easy - Global Edge)

Fly.io deploys containers globally for low latency.

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch (from project directory)
fly launch

# Set secrets (API keys)
fly secrets set OPENAI_API_KEY=sk-...
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly secrets set GOOGLE_API_KEY=AIza...
fly secrets set PUBMED_EMAIL=your@email.com

# Deploy
fly deploy
```

**Cost:** Free tier includes 3 shared VMs, ~$5/month for dedicated

---

## 🐳 Option 4: Docker (Self-Hosted)

For your own server (VPS, home server, etc.)

### Quick Start
```bash
# Clone the repo
git clone https://github.com/your-repo/reviewer.git
cd reviewer

# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

Access at `http://your-server-ip:8000`

### With HTTPS (Recommended)

Create `nginx.conf`:
```nginx
events { worker_connections 1024; }

http {
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/certs/fullchain.pem;
        ssl_certificate_key /etc/nginx/certs/privkey.pem;

        location / {
            proxy_pass http://reviewer:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

Update `docker-compose.yml` to include nginx (uncomment the nginx section).

---

## ☁️ Option 5: Cloud Run (Google Cloud)

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/ai-peer-review

# Deploy
gcloud run deploy ai-peer-review \
  --image gcr.io/YOUR_PROJECT/ai-peer-review \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "OPENAI_API_KEY=sk-..."
```

---

## 🔑 Environment Variables

Configure these in your deployment platform:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | One of these | OpenAI API key |
| `ANTHROPIC_API_KEY` | required | Anthropic/Claude API key |
| `GOOGLE_API_KEY` | | Google/Gemini API key |
| `PERPLEXITY_API_KEY` | Optional | Perplexity API key |
| `PUBMED_EMAIL` | Optional | Email for PubMed API |

**Note:** You need at least ONE AI provider key (OpenAI, Anthropic, or Google) for reviews to work. Users can also add keys via the Settings page in the UI.

---

## 💾 Persistent Storage

The app stores data in `/app/data/`:
- `reviews.db` - SQLite database with all reviews
- `settings.json` - User-configured API keys

For production, mount a persistent volume:

```yaml
# docker-compose.yml
volumes:
  - ./data:/app/data
```

For cloud platforms, use their persistent storage options.

---

## 🔒 Security Notes

1. **API Keys**: Never commit API keys to git. Use environment variables or the Settings page.

2. **HTTPS**: Always use HTTPS in production. Most platforms provide this automatically.

3. **Access Control**: The app currently has no authentication. For private use, add:
   - Basic auth via nginx
   - Cloud platform authentication (Railway, Render have this built-in)
   - Add your own auth middleware

---

## 📊 Resource Requirements

| Usage | RAM | CPU | Storage |
|-------|-----|-----|---------|
| Light (few reviews/day) | 512MB | 0.5 vCPU | 1GB |
| Medium (dozens/day) | 1GB | 1 vCPU | 5GB |
| Heavy (hundreds/day) | 2GB+ | 2 vCPU | 10GB+ |

The app is stateless except for the SQLite database, so it scales horizontally.

