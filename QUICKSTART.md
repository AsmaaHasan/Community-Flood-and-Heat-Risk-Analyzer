# 🚀 Quick Start Guide

Get the Community Flood and Heat Risk Analyzer running in 5 minutes!

## Prerequisites

- A Replit account (or any Python 3.11+ environment)
- Internet connection (for API data fetching)

## Step 1: Run the Application

### On Replit (Recommended)
1. Open this Repl in your browser
2. Click the green **Run** button at the top
3. Wait for the Streamlit server to start (10-15 seconds)
4. The application will open in the webview panel

### Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export DATABASE_URL="postgresql://user:password@host:port/database"

# Run the application
streamlit run app.py --server.port 5000
```

## Step 2: Configure Location

In the sidebar:
1. Select **Manila, Philippines** (default) or **Custom Coordinates**
2. For custom locations, enter latitude and longitude values
3. Click the **🔍 Analyze Risk** button

## Step 3: View Results

The dashboard displays:
- **Current Risk Assessment**: Flood and heat risk levels (Low/Medium/High)
- **Interactive Map**: Color-coded risk zones with pop-up details
- **Risk Trends**: 30-day historical analysis charts
- **NLP Insights**: News sentiment analysis results
- **Model Performance**: ML model accuracy and feature importance

## Step 4: Explore Advanced Features

### Enhanced NLP Analysis (Recommended)
- ✅ Check **"Use Enhanced NLP (TF-IDF + NER)"** in the sidebar
- This enables advanced text classification, urgency detection, and entity extraction

### Model Selection
Choose between:
- **Ensemble (RF + XGB)** - Most accurate (default)
- **Random Forest Only** - Fast baseline
- **XGBoost Only** - High performance gradient boosting

### Model Performance Dashboard
- ✅ Check **"Show Model Performance Comparison"**
- View accuracy metrics for both flood and heat models
- See which model performs best for each risk type

## Optional: Add API Keys

For enhanced real-time data (not required for demo):

1. **OpenWeatherMap API Key** (free tier)
   - Sign up at: https://openweathermap.org/api
   - Get your API key
   - Add to Replit Secrets: `OPENWEATHER_API_KEY`

2. **NewsAPI Key** (free tier)
   - Sign up at: https://newsapi.org/register
   - Get your API key
   - Add to Replit Secrets: `NEWSAPI_KEY`

**Without API keys**: The system uses realistic demonstration data automatically.

## Database Features (Optional)

The application includes a PostgreSQL database for:
- User accounts and authentication
- Saved monitoring locations
- Alert thresholds and notifications
- Historical risk data

To initialize the database:
```bash
python init_db.py
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for full database setup.

## Troubleshooting

### Application won't start
- Check that all packages are installed: `pip list`
- Restart the workflow: Click **Stop** then **Run**

### "No data available" message
- The system is using demo data (this is normal)
- Add API keys for real-time data (optional)

### Map not displaying
- Refresh the page
- Check browser console for errors (F12)
- Ensure Folium package is installed

### Database connection errors
- Run `python init_db.py` to initialize tables
- Check `DATABASE_URL` environment variable is set
- Verify PostgreSQL is running

## Next Steps

- Read [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production deployment
- Check [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for technical details
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute improvements

## Support

- 📖 Documentation: See all `.md` files in the project root
- 🐛 Issues: Check inline code comments for troubleshooting
- 💡 Questions: Review `replit.md` for architecture details
