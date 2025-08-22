import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Tab,
  Tabs,
  Card,
  CardContent,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
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

const ChurnCLVAnalysis: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);

  const churnData = [
    { segment: 'Champions', churnRate: 5, count: 1250 },
    { segment: 'Loyal', churnRate: 10, count: 2100 },
    { segment: 'Potential', churnRate: 15, count: 1800 },
    { segment: 'New', churnRate: 25, count: 3200 },
    { segment: 'At Risk', churnRate: 45, count: 980 },
    { segment: 'Lost', churnRate: 85, count: 450 },
  ];

  const clvDistribution = [
    { range: '$0-100', count: 2500, percentage: 25 },
    { range: '$100-500', count: 3500, percentage: 35 },
    { range: '$500-1000', count: 2000, percentage: 20 },
    { range: '$1000-5000', count: 1500, percentage: 15 },
    { range: '$5000+', count: 500, percentage: 5 },
  ];

  const cohortCLV = [
    { month: 'Jan', avgCLV: 120, newCustomers: 450 },
    { month: 'Feb', avgCLV: 135, newCustomers: 520 },
    { month: 'Mar', avgCLV: 142, newCustomers: 480 },
    { month: 'Apr', avgCLV: 155, newCustomers: 510 },
    { month: 'May', avgCLV: 168, newCustomers: 580 },
    { month: 'Jun', avgCLV: 175, newCustomers: 620 },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 4, fontWeight: 700 }}>
        Churn & Customer Lifetime Value Analysis
      </Typography>

      <Paper sx={{ borderRadius: 3 }}>
        <Tabs
          value={tabValue}
          onChange={(e, newValue) => setTabValue(newValue)}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Churn Analysis" />
          <Tab label="CLV Analysis" />
          <Tab label="Churn Prevention" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Churn Rate by Customer Segment
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={churnData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="segment" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar 
                      dataKey="churnRate" 
                      fill="#e74c3c" 
                      name="Churn Rate (%)"
                      radius={[8, 8, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>

            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Churn Risk Indicators
                </Typography>
                {[
                  { factor: 'Low Frequency', risk: 75, color: '#e74c3c' },
                  { factor: 'Low Value', risk: 60, color: '#f39c12' },
                  { factor: 'Single Purchase', risk: 85, color: '#c0392b' },
                  { factor: 'No Engagement', risk: 90, color: '#a93226' },
                ].map((factor) => (
                  <Box key={factor.factor} sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">{factor.factor}</Typography>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {factor.risk}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={factor.risk} 
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        backgroundColor: '#ecf0f1',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: factor.color,
                          borderRadius: 4,
                        },
                      }}
                    />
                  </Box>
                ))}
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  CLV Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={clvDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar 
                      dataKey="count" 
                      fill="#3498db" 
                      name="Customers"
                      radius={[8, 8, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Average CLV by Cohort
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={cohortCLV}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="avgCLV" 
                      stroke="#27ae60" 
                      fill="#27ae60" 
                      fillOpacity={0.3}
                      name="Average CLV ($)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            {[
              {
                title: 'High-Value at Risk',
                customers: 450,
                revenue: '$2.3M',
                action: 'Personalized retention offer',
                priority: 'high',
              },
              {
                title: 'Recent Churners',
                customers: 230,
                revenue: '$890K',
                action: 'Win-back campaign',
                priority: 'medium',
              },
              {
                title: 'Declining Engagement',
                customers: 680,
                revenue: '$1.5M',
                action: 'Re-engagement emails',
                priority: 'medium',
              },
              {
                title: 'Single Purchase',
                customers: 1200,
                revenue: '$450K',
                action: 'Second purchase incentive',
                priority: 'low',
              },
            ].map((segment) => (
              <Grid item xs={12} md={6} key={segment.title}>
                <Card sx={{ borderRadius: 2 }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        {segment.title}
                      </Typography>
                      <Chip 
                        label={segment.priority} 
                        size="small"
                        color={
                          segment.priority === 'high' ? 'error' : 
                          segment.priority === 'medium' ? 'warning' : 'default'
                        }
                      />
                    </Box>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Customers
                        </Typography>
                        <Typography variant="h5">
                          {segment.customers.toLocaleString()}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Revenue at Risk
                        </Typography>
                        <Typography variant="h5">
                          {segment.revenue}
                        </Typography>
                      </Grid>
                    </Grid>
                    <Box sx={{ mt: 2, p: 2, backgroundColor: '#f8f9fa', borderRadius: 1 }}>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        Recommended Action:
                      </Typography>
                      <Typography variant="body2">
                        {segment.action}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default ChurnCLVAnalysis;