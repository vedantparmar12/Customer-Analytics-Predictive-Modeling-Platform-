# 🚀 Frontend Setup Guide - E-commerce Analytics Platform

## Prerequisites
- Node.js (v14 or higher)
- npm or yarn
- Python backend running (optional for full functionality)

## Step 1: Install Frontend Dependencies

### On Windows:
```bash
cd frontend
npm install
```

### On Linux/Mac:
```bash
cd frontend
npm install
```

## Step 2: Start the Backend API (Optional but Recommended)

Open a new terminal and run:
```bash
# From project root directory
python -m uvicorn api.main:app --reload --port 8000
```

If you get an error about missing modules, install FastAPI:
```bash
pip install fastapi uvicorn redis
```

## Step 3: Start the React Frontend

In the frontend directory:
```bash
npm start
```

This will:
- Start the development server
- Open your browser at http://localhost:3000
- Hot reload on any code changes

## Step 4: Access the Application

The frontend will automatically open in your browser. You'll see:

### Main Features:
1. **Executive Dashboard** - KPIs, revenue charts, customer metrics
2. **Advanced Segmentation** - Customer segments, RFM analysis
3. **Churn & CLV Analysis** - Predictive models, risk assessment
4. **A/B Testing Suite** - Sample size calculator, test results
5. **Cohort Analysis** - Retention matrices, LTV by cohort
6. **Recommendation Engine** - Product recommendations
7. **Business Insights** - Key metrics and actionable insights

## Alternative: Run Without Backend

If you want to run just the frontend with mock data:

1. The frontend will still work but will show sample/cached data
2. API calls will fail gracefully
3. You can still explore the UI and visualizations

## Troubleshooting

### Issue: npm install fails
```bash
# Clear npm cache
npm cache clean --force
# Try again
npm install
```

### Issue: Port 3000 already in use
```bash
# On Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# On Linux/Mac
lsof -i :3000
kill -9 <PID>
```

### Issue: Backend connection fails
- Check if backend is running on port 8000
- Check the proxy setting in package.json (should be "http://localhost:8000")

## Build for Production

To create a production build:
```bash
npm run build
```

This creates an optimized build in the `build/` folder.

## Using the Application

### With Real Data:
1. Run the pipeline first: `python pipeline/master_pipeline.py`
2. Start the API: `python -m uvicorn api.main:app --reload`
3. Start the frontend: `cd frontend && npm start`

### Quick Start (Frontend Only):
```bash
cd frontend
npm install
npm start
```

## Available Scripts

- `npm start` - Run development server
- `npm test` - Run tests
- `npm run build` - Create production build
- `npm run eject` - Eject from Create React App (not recommended)

## Success\! 
Your browser should now show the Customer Analytics Platform at http://localhost:3000
