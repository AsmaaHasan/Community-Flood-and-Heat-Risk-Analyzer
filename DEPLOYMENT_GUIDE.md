# 🚀 Deployment Guide

Complete guide for deploying the Community Flood and Heat Risk Analyzer to production.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Database Configuration](#database-configuration)
4. [API Keys Configuration](#api-keys-configuration)
5. [Deployment Options](#deployment-options)
6. [Production Checklist](#production-checklist)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 1GB for application + database
- **CPU**: 2 cores recommended
- **Internet**: Required for API data fetching

### Software Dependencies
All Python packages are listed in `pyproject.toml`:
- Streamlit 1.51.0+
- Scikit-learn 1.7.2+
- XGBoost 3.1.1+
- SQLAlchemy 2.0.44+
- PostgreSQL 14+ (for database features)
- See `pyproject.toml` for complete list

## Environment Setup

### 1. Clone or Fork the Repository

On Replit:
- Fork this Repl to your account
- All dependencies auto-install

Locally:
```bash
git clone <repository-url>
cd community-flood-heat-analyzer
```

### 2. Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Using uv (Replit default):
```bash
uv pip install -r requirements.txt
```

### 3. Download NLTK Data

The application automatically downloads required NLTK data on first run:
- `punkt` - Tokenizer
- `stopwords` - Stop word lists
- `averaged_perceptron_tagger` - POS tagger
- `maxent_ne_chunker` - Named entity chunker
- `words` - Word lists

Manual download (if needed):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

## Database Configuration

### PostgreSQL Setup

#### On Replit
1. Database is pre-configured with environment variables:
   - `DATABASE_URL`
   - `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`

2. Initialize the database:
```bash
python init_db.py
```

3. Verify tables were created:
```bash
python -c "from database import get_database_engine; from sqlalchemy import inspect; print(inspect(get_database_engine()).get_table_names())"
```

#### On Other Platforms

1. Install PostgreSQL 14+
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql
```

2. Create database and user:
```sql
CREATE DATABASE flood_heat_analyzer;
CREATE USER analyzer_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE flood_heat_analyzer TO analyzer_user;
```

3. Set environment variable:
```bash
export DATABASE_URL="postgresql://analyzer_user:secure_password@localhost:5432/flood_heat_analyzer"
```

4. Initialize tables:
```bash
python init_db.py
```

### Database Schema

The system creates 5 tables:

1. **users** - User accounts and authentication
2. **saved_locations** - Monitored locations per user
3. **alert_thresholds** - User-defined alert criteria
4. **risk_history** - Historical risk predictions
5. **alert_logs** - Notification audit trail

See `database.py` for complete schema definitions.

## API Keys Configuration

### Required APIs (Optional but Recommended)

#### 1. OpenWeatherMap API
- **Purpose**: Real-time weather data and forecasts
- **Free Tier**: 1,000 calls/day
- **Sign up**: https://openweathermap.org/api
- **Variable name**: `OPENWEATHER_API_KEY`

#### 2. NewsAPI
- **Purpose**: News article collection for sentiment analysis
- **Free Tier**: 100 requests/day, 1 month history
- **Sign up**: https://newsapi.org/register
- **Variable name**: `NEWSAPI_KEY`

### Adding Secrets to Replit

1. Open the **Secrets** tab (lock icon in left sidebar)
2. Add each key:
   - Key: `OPENWEATHER_API_KEY`
   - Value: Your API key
   - Click **Add secret**
3. Repeat for `NEWSAPI_KEY`
4. Restart the application

### Adding Environment Variables Locally

Create a `.env` file:
```bash
# API Keys (optional - system uses demo data if not provided)
OPENWEATHER_API_KEY=your_openweather_key_here
NEWSAPI_KEY=your_newsapi_key_here

# Database (required for user features)
DATABASE_URL=postgresql://user:password@localhost:5432/flood_heat_analyzer

# Session Secret (required for authentication)
SESSION_SECRET=your_random_secret_key_here
```

Load environment variables:
```bash
# Using python-dotenv (included)
python app.py  # Automatically loads .env

# Or manually
export $(cat .env | xargs)
```

## Deployment Options

### Option 1: Replit Deployment (Recommended for Quick Start)

**Autoscale Deployment**:
1. Click **Deploy** button in Replit
2. Select **Autoscale** deployment type
3. Configure:
   - Port: 5000 (pre-configured)
   - Build command: `pip install -r requirements.txt`
   - Run command: `streamlit run app.py --server.port 5000`
4. Click **Deploy**
5. Your app will be live at `https://your-repl-name.your-username.repl.co`

**Reserved VM Deployment** (For always-on production):
1. Click **Deploy** → **Reserved VM**
2. Choose VM size (2GB RAM minimum recommended)
3. Configure same as above
4. Higher reliability and consistent performance

### Option 2: Standalone Server Deployment

#### Using systemd (Linux)

1. Create service file `/etc/systemd/system/flood-analyzer.service`:
```ini
[Unit]
Description=Community Flood and Heat Risk Analyzer
After=network.target postgresql.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/flood-analyzer
Environment="PATH=/opt/flood-analyzer/venv/bin"
Environment="DATABASE_URL=postgresql://user:pass@localhost/flood_heat_analyzer"
ExecStart=/opt/flood-analyzer/venv/bin/streamlit run app.py --server.port 5000 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

2. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable flood-analyzer
sudo systemctl start flood-analyzer
```

#### Using Docker

1. Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Expose port
EXPOSE 5000

# Run application
CMD ["streamlit", "run", "app.py", "--server.port", "5000", "--server.address", "0.0.0.0"]
```

2. Build and run:
```bash
docker build -t flood-analyzer .
docker run -p 5000:5000 \
  -e DATABASE_URL="postgresql://..." \
  -e OPENWEATHER_API_KEY="..." \
  -e NEWSAPI_KEY="..." \
  flood-analyzer
```

### Option 3: Cloud Platform Deployment

#### Heroku
```bash
# Install Heroku CLI
# Create Procfile:
echo "web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
heroku addons:create heroku-postgresql:mini
heroku config:set OPENWEATHER_API_KEY=xxx
heroku config:set NEWSAPI_KEY=xxx
git push heroku main
```

#### Railway
1. Connect GitHub repository
2. Add PostgreSQL database
3. Set environment variables in Railway dashboard
4. Deploy automatically on push

#### DigitalOcean App Platform
1. Create new app from GitHub
2. Add managed PostgreSQL database
3. Configure environment variables
4. Set build command: `pip install -r requirements.txt`
5. Set run command: `streamlit run app.py --server.port 5000`

## Production Checklist

### Security
- [ ] Change default `SESSION_SECRET` to a strong random value
- [ ] Store API keys in environment variables (never commit to Git)
- [ ] Enable HTTPS/TLS for production deployment
- [ ] Set up firewall rules to restrict database access
- [ ] Implement rate limiting for API endpoints
- [ ] Enable CORS only for trusted domains

### Performance
- [ ] Configure database connection pooling (default: 5 connections, max 10)
- [ ] Set Streamlit cache TTL appropriately (default: 600s)
- [ ] Monitor memory usage (XGBoost + data can use 1-2GB)
- [ ] Enable gzip compression for static assets
- [ ] Configure CDN for static assets (if applicable)

### Monitoring
- [ ] Set up application logging
- [ ] Monitor API rate limits and usage
- [ ] Track database query performance
- [ ] Set up health check endpoint
- [ ] Configure alerts for downtime or errors

### Data
- [ ] Schedule regular database backups
- [ ] Set up log rotation
- [ ] Configure data retention policies
- [ ] Test disaster recovery procedures

## Monitoring and Maintenance

### Application Health Check

Create `health_check.py`:
```python
import requests
import sys

try:
    response = requests.get('http://localhost:5000/_stcore/health', timeout=5)
    if response.status_code == 200:
        print("✅ Application is healthy")
        sys.exit(0)
    else:
        print(f"❌ Health check failed: {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Health check failed: {e}")
    sys.exit(1)
```

### Database Backup

```bash
# Backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore
psql $DATABASE_URL < backup_20250131_120000.sql
```

### Log Monitoring

Application logs locations:
- Streamlit: `~/.streamlit/logs/`
- Application: stdout/stderr (capture with systemd or Docker)

### Performance Monitoring

Key metrics to track:
- Response time for risk predictions
- API call success rate
- Database query duration
- Memory usage
- Cache hit rate

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError"
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

#### "Database connection failed"
```bash
# Verify DATABASE_URL is set
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1"

# Reinitialize tables
python init_db.py
```

#### "NLTK data not found"
```bash
# Download NLTK data manually
python -c "import nltk; nltk.download('all')"
```

#### High memory usage
- Reduce Streamlit cache TTL
- Decrease database connection pool size
- Use Random Forest only (lighter than ensemble)
- Disable model comparison dashboard

#### Slow prediction performance
- Enable Enhanced NLP only when needed
- Use XGBoost only (faster than ensemble)
- Increase cache TTL to reduce API calls
- Optimize database queries with indexes

### Getting Help

1. Check application logs for error details
2. Review `replit.md` for architecture understanding
3. See inline code comments in source files
4. Check GitHub issues (if open source)

## Next Steps

After deployment:
1. Test all features in production environment
2. Set up monitoring and alerts
3. Configure automated backups
4. Plan for scaling (if needed)
5. Document any custom configurations

For development and contributions, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).
