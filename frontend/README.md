# Customer Analytics Platform - React Frontend

A professional, enterprise-grade React frontend for the Customer Analytics Platform, replacing the Streamlit interface with a modern, responsive, and feature-rich UI.

## Features

### 📊 Executive Dashboard
- Real-time KPI monitoring with animated cards
- Revenue and growth trend visualizations
- Customer distribution analysis
- Profitability metrics by segment
- Business insights with actionable recommendations

### 🎯 Advanced Segmentation
- RFM (Recency, Frequency, Monetary) segmentation
- CLV-based customer tiers
- Behavioral clustering
- Geographic analysis
- Interactive visualizations (scatter plots, treemaps, pie charts)

### 💰 Churn & CLV Analysis
- Churn risk assessment by segment
- Customer lifetime value distribution
- Churn prevention strategies
- Win-back campaign targeting
- Predictive analytics integration

### 🧪 A/B Testing Suite
- Sample size calculator with real-time updates
- Test results analyzer with statistical significance
- Sensitivity analysis
- Sequential testing support
- Bayesian probability distributions

### 📈 Cohort Analysis
- Customer retention heatmaps
- Revenue retention tracking
- LTV progression analysis
- Cohort quality metrics
- CAC payback period visualization

### 🤖 Recommendation Engine
- Cross-sell opportunity identification
- Upsell potential analysis
- Win-back campaign management
- Personalized recommendations
- Revenue uplift calculations

### 💡 Business Insights
- Categorized insights and recommendations
- Priority-based action items
- Impact assessment
- Interactive accordion view
- Real-time revenue impact tracking

## Tech Stack

- **React 18** with TypeScript
- **Material-UI (MUI)** for component library
- **Redux Toolkit** for state management
- **React Router** for navigation
- **Recharts** for data visualization
- **Axios** for API integration
- **Chart.js** for advanced charts

## Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Dashboard/      # Dashboard components
│   │   └── Layout/         # Layout components (Sidebar, Header)
│   ├── pages/             # Page components
│   ├── services/          # API services
│   ├── store/             # Redux store and slices
│   ├── styles/            # Theme and global styles
│   ├── types/             # TypeScript type definitions
│   ├── utils/             # Utility functions
│   ├── App.tsx           # Main app component
│   └── index.tsx         # Entry point
└── package.json
```

## Design Principles

### 1. **Professional UI/UX**
- Clean, modern design with gradient accents
- Consistent color scheme aligned with brand
- Smooth animations and transitions
- Interactive hover effects

### 2. **Responsive Design**
- Mobile-first approach
- Adaptive layouts for all screen sizes
- Touch-friendly interfaces
- Optimized performance

### 3. **Data Visualization**
- Interactive charts with tooltips
- Real-time data updates
- Multiple visualization options
- Export capabilities

### 4. **User Experience**
- Intuitive navigation
- Loading states and error handling
- Contextual help and tooltips
- Keyboard shortcuts support

## API Integration

The frontend integrates with the FastAPI backend through:
- RESTful API endpoints
- JWT authentication
- Real-time data fetching
- Error handling and retry logic

## Available Scripts

```bash
# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test

# Eject from Create React App
npm run eject
```

## Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000/api
REACT_APP_ENV=development
```

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance Optimizations

- Code splitting with React.lazy()
- Memoization with React.memo()
- Virtual scrolling for large datasets
- Image lazy loading
- Bundle size optimization

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License