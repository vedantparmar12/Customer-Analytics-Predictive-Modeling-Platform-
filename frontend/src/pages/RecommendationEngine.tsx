import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Card,
  CardContent,
  Tab,
  Tabs,
  Chip,
  Button,
  LinearProgress,
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const RecommendationEngine: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);

  const crossSellData = [
    { segment: 'Low', customers: 2500, revenue: 125000, potential: 250000 },
    { segment: 'Medium', customers: 3200, revenue: 480000, potential: 720000 },
    { segment: 'High', customers: 1800, revenue: 540000, potential: 675000 },
    { segment: 'VIP', customers: 500, revenue: 350000, potential: 420000 },
  ];

  const upsellData = [
    { segment: 'Bottom 25%', current: 45, target: 125, potential: 80 },
    { segment: 'Middle 50%', current: 185, target: 350, potential: 165 },
    { segment: 'Top 25%', current: 650, target: 780, potential: 130 },
  ];

  const winBackSegments = [
    { name: 'High-Value Lost', value: 450, color: '#e74c3c' },
    { name: 'Recent Churners', value: 680, color: '#f39c12' },
    { name: 'Single Purchase', value: 1200, color: '#3498db' },
  ];

  const COLORS = ['#e74c3c', '#f39c12', '#27ae60', '#9b59b6'];

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 4, fontWeight: 700 }}>
        Recommendation Engine
      </Typography>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#f8f9fa' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Total Customers
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#667eea' }}>
                98,765
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#f8f9fa' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Avg Orders per Customer
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#27ae60' }}>
                3.4
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#f8f9fa' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Avg Revenue per Customer
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#3498db' }}>
                $245
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#f8f9fa' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Revenue Uplift Potential
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#f39c12' }}>
                $4.2M
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper sx={{ borderRadius: 3 }}>
        <Tabs
          value={tabValue}
          onChange={(e, newValue) => setTabValue(newValue)}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Cross-Sell Opportunities" />
          <Tab label="Upsell Potential" />
          <Tab label="Win-Back Campaigns" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Cross-Sell by Purchase Frequency
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={crossSellData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="segment" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="revenue" fill="#3498db" name="Current Revenue" />
                    <Bar dataKey="potential" fill="#27ae60" name="Potential Revenue" />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Recommendations
                </Typography>
                <Box sx={{ mt: 2 }}>
                  {[
                    { segment: 'Low Frequency', action: 'Product bundles', impact: '+15%' },
                    { segment: 'Medium Frequency', action: 'Complementary items', impact: '+25%' },
                    { segment: 'High Frequency', action: 'Loyalty rewards', impact: '+10%' },
                    { segment: 'VIP Customers', action: 'Premium services', impact: '+20%' },
                  ].map((rec) => (
                    <Card key={rec.segment} sx={{ mb: 2 }}>
                      <CardContent>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                          {rec.segment}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {rec.action}
                        </Typography>
                        <Chip 
                          label={`Revenue ${rec.impact}`} 
                          size="small" 
                          color="success" 
                          sx={{ mt: 1 }}
                        />
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Upsell Potential Analysis
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={upsellData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="segment" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="current" fill="#3498db" name="Current Avg ($)" />
                    <Bar dataKey="target" fill="#27ae60" name="Target Avg ($)" />
                    <Bar dataKey="potential" fill="#f39c12" name="Uplift Potential ($)" />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Total Upsell Potential
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 700, color: '#27ae60', mb: 3 }}>
                  $2.8M
                </Typography>
                <Box>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Implementation Progress
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={35} 
                    sx={{ 
                      height: 10, 
                      borderRadius: 5,
                      backgroundColor: '#ecf0f1',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#27ae60',
                        borderRadius: 5,
                      }
                    }}
                  />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    35% of customers engaged
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Win-Back Priority Segments
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <PieChart>
                    <Pie
                      data={winBackSegments}
                      cx="50%"
                      cy="50%"
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {winBackSegments.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Card sx={{ borderLeft: '4px solid #e74c3c' }}>
                    <CardContent>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Churned Customers
                      </Typography>
                      <Typography variant="h3" sx={{ color: '#e74c3c', my: 2 }}>
                        2,330
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Lost Revenue: $570K
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12}>
                  <Card sx={{ borderLeft: '4px solid #27ae60' }}>
                    <CardContent>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Win-Back Potential
                      </Typography>
                      <Typography variant="h3" sx={{ color: '#27ae60', my: 2 }}>
                        $285K
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        50% recovery rate target
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12}>
                  <Button variant="contained" fullWidth size="large">
                    Launch Win-Back Campaign
                  </Button>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default RecommendationEngine;