import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Paper,
  Chip,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Treemap,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';

const AdvancedSegmentation: React.FC = () => {
  const [segmentationType, setSegmentationType] = useState('RFM');
  const [visualization, setVisualization] = useState('scatter');

  const rfmData = [
    { name: 'Champions', customers: 1250, revenue: 3200000, avgOrders: 12, color: '#9b59b6' },
    { name: 'Loyal Customers', customers: 2100, revenue: 2100000, avgOrders: 8, color: '#3498db' },
    { name: 'Potential Loyalists', customers: 1800, revenue: 1500000, avgOrders: 5, color: '#27ae60' },
    { name: 'New Customers', customers: 3200, revenue: 800000, avgOrders: 2, color: '#f39c12' },
    { name: 'At Risk', customers: 980, revenue: 600000, avgOrders: 4, color: '#e74c3c' },
  ];

  const scatterData = Array.from({ length: 100 }, (_, i) => ({
    frequency: Math.random() * 20,
    monetary: Math.random() * 5000,
    recency: Math.random() * 365,
    segment: rfmData[Math.floor(Math.random() * rfmData.length)].name,
  }));

  const getSegmentColor = (segment: string) => {
    const seg = rfmData.find(s => s.name === segment);
    return seg ? seg.color : '#95a5a6';
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 4, fontWeight: 700 }}>
        Advanced Customer Segmentation
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Segmentation Type</InputLabel>
            <Select
              value={segmentationType}
              onChange={(e) => setSegmentationType(e.target.value)}
              label="Segmentation Type"
            >
              <MenuItem value="RFM">RFM Segmentation</MenuItem>
              <MenuItem value="CLV">CLV Tiers</MenuItem>
              <MenuItem value="Behavioral">Behavioral Clusters</MenuItem>
              <MenuItem value="Geographic">Geographic Analysis</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Visualization</InputLabel>
            <Select
              value={visualization}
              onChange={(e) => setVisualization(e.target.value)}
              label="Visualization"
            >
              <MenuItem value="scatter">Scatter Plot</MenuItem>
              <MenuItem value="treemap">Treemap</MenuItem>
              <MenuItem value="pie">Pie Chart</MenuItem>
              <MenuItem value="table">Table View</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              {segmentationType} Analysis
            </Typography>
            
            {visualization === 'scatter' && (
              <ResponsiveContainer width="100%" height={500}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="frequency" name="Frequency" unit=" orders" />
                  <YAxis dataKey="monetary" name="Monetary Value" unit="$" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Legend />
                  {rfmData.map((segment) => (
                    <Scatter
                      key={segment.name}
                      name={segment.name}
                      data={scatterData.filter(d => d.segment === segment.name)}
                      fill={segment.color}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            )}

            {visualization === 'pie' && (
              <ResponsiveContainer width="100%" height={500}>
                <PieChart>
                  <Pie
                    data={rfmData}
                    cx="50%"
                    cy="50%"
                    outerRadius={150}
                    fill="#8884d8"
                    dataKey="customers"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {rfmData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            )}

            {visualization === 'table' && (
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Segment</TableCell>
                      <TableCell align="right">Customers</TableCell>
                      <TableCell align="right">Revenue</TableCell>
                      <TableCell align="right">Avg Orders</TableCell>
                      <TableCell align="right">% of Total</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {rfmData.map((row) => (
                      <TableRow key={row.name}>
                        <TableCell>
                          <Chip 
                            label={row.name} 
                            sx={{ backgroundColor: row.color, color: 'white' }}
                          />
                        </TableCell>
                        <TableCell align="right">{row.customers.toLocaleString()}</TableCell>
                        <TableCell align="right">${row.revenue.toLocaleString()}</TableCell>
                        <TableCell align="right">{row.avgOrders}</TableCell>
                        <TableCell align="right">
                          {((row.customers / rfmData.reduce((a, b) => a + b.customers, 0)) * 100).toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, borderRadius: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Segment Performance
            </Typography>
            {rfmData.map((segment) => (
              <Card key={segment.name} sx={{ mb: 2, borderLeft: `4px solid ${segment.color}` }}>
                <CardContent>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, color: segment.color }}>
                    {segment.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Customers: {segment.customers.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Revenue: ${(segment.revenue / 1000000).toFixed(1)}M
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Avg Orders: {segment.avgOrders}
                  </Typography>
                </CardContent>
              </Card>
            ))}
          </Paper>

          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Actions
            </Typography>
            <Button 
              variant="contained" 
              fullWidth 
              sx={{ mb: 2 }}
            >
              Export Segments
            </Button>
            <Button 
              variant="outlined" 
              fullWidth
            >
              Create Campaign
            </Button>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AdvancedSegmentation;