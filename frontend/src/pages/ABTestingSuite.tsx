import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  TextField,
  Button,
  Card,
  CardContent,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Divider,
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

const ABTestingSuite: React.FC = () => {
  const [testType, setTestType] = useState('conversion');
  const [baselineRate, setBaselineRate] = useState(5);
  const [mde, setMDE] = useState(20);
  const [confidence, setConfidence] = useState(95);
  const [power, setPower] = useState(80);
  const [sampleSize, setSampleSize] = useState<number | null>(null);

  const calculateSampleSize = () => {
    const z_alpha = 1.96;
    const z_beta = 0.84;
    const p1 = baselineRate / 100;
    const p2 = p1 * (1 + mde / 100);
    const p_avg = (p1 + p2) / 2;
    
    const n = Math.ceil(
      (2 * p_avg * (1 - p_avg) * Math.pow(z_alpha + z_beta, 2)) /
      Math.pow(p2 - p1, 2)
    );
    
    setSampleSize(n);
  };

  const sensitivityData = Array.from({ length: 10 }, (_, i) => {
    const mdeValue = 5 + i * 5;
    const p1 = baselineRate / 100;
    const p2 = p1 * (1 + mdeValue / 100);
    const p_avg = (p1 + p2) / 2;
    const z_alpha = 1.96;
    const z_beta = 0.84;
    
    const n = Math.ceil(
      (2 * p_avg * (1 - p_avg) * Math.pow(z_alpha + z_beta, 2)) /
      Math.pow(p2 - p1, 2)
    );
    
    return { mde: mdeValue, sampleSize: n };
  });

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 4, fontWeight: 700 }}>
        A/B Testing Suite
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Sample Size Calculator
            </Typography>
            
            <Grid container spacing={3} sx={{ mt: 1 }}>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Test Type</InputLabel>
                  <Select
                    value={testType}
                    onChange={(e) => setTestType(e.target.value)}
                    label="Test Type"
                  >
                    <MenuItem value="conversion">Conversion Rate</MenuItem>
                    <MenuItem value="revenue">Revenue (ARPU)</MenuItem>
                    <MenuItem value="retention">Retention</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12}>
                <Typography gutterBottom>
                  Baseline Rate: {baselineRate}%
                </Typography>
                <Slider
                  value={baselineRate}
                  onChange={(e, v) => setBaselineRate(v as number)}
                  min={0.1}
                  max={50}
                  step={0.1}
                  valueLabelDisplay="auto"
                />
              </Grid>

              <Grid item xs={12}>
                <Typography gutterBottom>
                  Minimum Detectable Effect: {mde}%
                </Typography>
                <Slider
                  value={mde}
                  onChange={(e, v) => setMDE(v as number)}
                  min={1}
                  max={100}
                  valueLabelDisplay="auto"
                />
              </Grid>

              <Grid item xs={12}>
                <Typography gutterBottom>
                  Confidence Level: {confidence}%
                </Typography>
                <Slider
                  value={confidence}
                  onChange={(e, v) => setConfidence(v as number)}
                  min={80}
                  max={99}
                  valueLabelDisplay="auto"
                />
              </Grid>

              <Grid item xs={12}>
                <Typography gutterBottom>
                  Statistical Power: {power}%
                </Typography>
                <Slider
                  value={power}
                  onChange={(e, v) => setPower(v as number)}
                  min={70}
                  max={95}
                  valueLabelDisplay="auto"
                />
              </Grid>

              <Grid item xs={12}>
                <Button 
                  variant="contained" 
                  fullWidth 
                  size="large"
                  onClick={calculateSampleSize}
                >
                  Calculate Sample Size
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12} lg={6}>
          {sampleSize && (
            <Paper sx={{ p: 3, borderRadius: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                Results
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={6}>
                  <Card sx={{ backgroundColor: '#f8f9fa' }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">
                        Sample Size per Variant
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: '#667eea' }}>
                        {sampleSize.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card sx={{ backgroundColor: '#f8f9fa' }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">
                        Total Sample Size
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: '#27ae60' }}>
                        {(sampleSize * 2).toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12}>
                  <Card sx={{ backgroundColor: '#fff3cd', borderLeft: '4px solid #f39c12' }}>
                    <CardContent>
                      <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                        Estimated Duration
                      </Typography>
                      <Typography variant="body1">
                        With 5,000 daily visitors: {Math.ceil(sampleSize / 2500)} days
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Includes 20% buffer for variations in traffic
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          )}

          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Sample Size Sensitivity
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={sensitivityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mde" label={{ value: 'MDE (%)', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Sample Size', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="sampleSize" 
                  stroke="#667eea" 
                  strokeWidth={3}
                  dot={{ fill: '#667eea', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              Test Results Analyzer
            </Typography>
            
            <Grid container spacing={3} sx={{ mt: 1 }}>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Control Visitors"
                  type="number"
                  fullWidth
                  defaultValue={5000}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Control Conversions"
                  type="number"
                  fullWidth
                  defaultValue={250}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Treatment Visitors"
                  type="number"
                  fullWidth
                  defaultValue={5000}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="Treatment Conversions"
                  type="number"
                  fullWidth
                  defaultValue={300}
                />
              </Grid>
              <Grid item xs={12}>
                <Button variant="outlined" fullWidth size="large">
                  Analyze Results
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ABTestingSuite;