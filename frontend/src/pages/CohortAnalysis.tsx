import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
} from '@mui/material';

const CohortAnalysis: React.FC = () => {
  const [metric, setMetric] = useState('retention');
  const [displayType, setDisplayType] = useState('heatmap');
  const [cohortLimit, setCohortLimit] = useState(12);

  const generateCohortData = () => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const data = [];
    
    for (let i = 0; i < cohortLimit; i++) {
      const row = {
        cohort: months[i],
        month0: 100,
        month1: Math.max(70 - i * 2, 40) + Math.random() * 10,
        month2: Math.max(60 - i * 2, 30) + Math.random() * 10,
        month3: Math.max(50 - i * 2, 25) + Math.random() * 10,
        month4: Math.max(45 - i * 2, 20) + Math.random() * 10,
        month5: Math.max(40 - i * 2, 18) + Math.random() * 10,
      };
      data.push(row);
    }
    return data;
  };

  const cohortData = generateCohortData();

  const getColor = (value: number) => {
    if (value >= 80) return '#27ae60';
    if (value >= 60) return '#52c77e';
    if (value >= 40) return '#f39c12';
    if (value >= 20) return '#e67e22';
    return '#e74c3c';
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 4, fontWeight: 700 }}>
        Advanced Cohort Analysis
      </Typography>

      <Paper sx={{ p: 3, borderRadius: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Metric</InputLabel>
              <Select
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                label="Metric"
              >
                <MenuItem value="retention">Customer Retention</MenuItem>
                <MenuItem value="revenue">Revenue Retention</MenuItem>
                <MenuItem value="orders">Order Frequency</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Display Type</InputLabel>
              <Select
                value={displayType}
                onChange={(e) => setDisplayType(e.target.value)}
                label="Display Type"
              >
                <MenuItem value="heatmap">Heatmap</MenuItem>
                <MenuItem value="curves">Curves</MenuItem>
                <MenuItem value="waterfall">Waterfall</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography gutterBottom>
              Number of Cohorts: {cohortLimit}
            </Typography>
            <Slider
              value={cohortLimit}
              onChange={(e, v) => setCohortLimit(v as number)}
              min={5}
              max={12}
              valueLabelDisplay="auto"
            />
          </Grid>
        </Grid>
      </Paper>

      <Paper sx={{ p: 3, borderRadius: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
          {metric === 'retention' ? 'Customer' : metric === 'revenue' ? 'Revenue' : 'Order'} Retention by Cohort
        </Typography>
        
        {displayType === 'heatmap' && (
          <Box sx={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 20 }}>
              <thead>
                <tr>
                  <th style={{ padding: 12, textAlign: 'left', borderBottom: '2px solid #e0e0e0' }}>
                    Cohort
                  </th>
                  {[0, 1, 2, 3, 4, 5].map(month => (
                    <th key={month} style={{ padding: 12, textAlign: 'center', borderBottom: '2px solid #e0e0e0' }}>
                      Month {month}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cohortData.map((row, idx) => (
                  <tr key={idx}>
                    <td style={{ padding: 12, fontWeight: 600, borderBottom: '1px solid #f0f0f0' }}>
                      {row.cohort}
                    </td>
                    {['month0', 'month1', 'month2', 'month3', 'month4', 'month5'].map(month => {
                      const value = row[month as keyof typeof row] as number;
                      return (
                        <td 
                          key={month}
                          style={{ 
                            padding: 12, 
                            textAlign: 'center',
                            backgroundColor: getColor(value),
                            color: 'white',
                            fontWeight: 600,
                            borderBottom: '1px solid #f0f0f0'
                          }}
                        >
                          {value.toFixed(1)}%
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>
        )}
      </Paper>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Cohort Quality Analysis
            </Typography>
            <Grid container spacing={2}>
              {[
                { label: 'Best Performing', value: 'March', color: '#27ae60' },
                { label: 'Worst Performing', value: 'November', color: '#e74c3c' },
                { label: 'Avg 3-Month Retention', value: '45.2%', color: '#3498db' },
                { label: 'Avg 6-Month Retention', value: '32.8%', color: '#9b59b6' },
              ].map((stat) => (
                <Grid item xs={6} key={stat.label}>
                  <Box 
                    sx={{ 
                      p: 2, 
                      borderRadius: 2, 
                      backgroundColor: '#f8f9fa',
                      borderLeft: `4px solid ${stat.color}`
                    }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      {stat.label}
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: stat.color }}>
                      {stat.value}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Lifetime Value Analysis
            </Typography>
            <Grid container spacing={2}>
              {[
                { label: 'Avg 6-Month LTV', value: '$245', color: '#667eea' },
                { label: 'Avg 12-Month LTV', value: '$420', color: '#764ba2' },
                { label: 'Payback Period', value: '4.2 months', color: '#f39c12' },
                { label: 'LTV:CAC Ratio', value: '3.2x', color: '#27ae60' },
              ].map((stat) => (
                <Grid item xs={6} key={stat.label}>
                  <Box 
                    sx={{ 
                      p: 2, 
                      borderRadius: 2, 
                      backgroundColor: '#f8f9fa',
                      borderLeft: `4px solid ${stat.color}`
                    }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      {stat.label}
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: stat.color }}>
                      {stat.value}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default CohortAnalysis;