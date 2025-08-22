import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from 'recharts';
import { Box } from '@mui/material';

const ProfitabilityChart: React.FC = () => {
  const data = [
    { segment: 'VIP', revenue: 3200000, profit: 1280000, margin: 40 },
    { segment: 'High Value', revenue: 2100000, profit: 735000, margin: 35 },
    { segment: 'Medium Value', revenue: 1500000, profit: 450000, margin: 30 },
    { segment: 'Low Value', revenue: 800000, profit: 200000, margin: 25 },
    { segment: 'New Customers', revenue: 600000, profit: 120000, margin: 20 },
  ];

  const formatCurrency = (value: number) => {
    return `$${(value / 1000000).toFixed(1)}M`;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box
          sx={{
            background: 'white',
            border: '1px solid #e0e0e0',
            borderRadius: 1,
            p: 1.5,
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          }}
        >
          <p style={{ margin: 0, fontWeight: 600 }}>{label}</p>
          <p style={{ margin: '4px 0', color: '#3498db' }}>
            Revenue: {formatCurrency(payload[0].value)}
          </p>
          <p style={{ margin: '4px 0', color: '#27ae60' }}>
            Profit: {formatCurrency(payload[1].value)}
          </p>
          <p style={{ margin: '4px 0', color: '#e74c3c' }}>
            Margin: {payload[2].value}%
          </p>
        </Box>
      );
    }
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis 
          dataKey="segment" 
          tick={{ fill: '#7f8c8d' }}
          axisLine={{ stroke: '#e0e0e0' }}
        />
        <YAxis 
          yAxisId="left"
          tick={{ fill: '#7f8c8d' }}
          axisLine={{ stroke: '#e0e0e0' }}
          tickFormatter={formatCurrency}
        />
        <YAxis 
          yAxisId="right" 
          orientation="right"
          tick={{ fill: '#7f8c8d' }}
          axisLine={{ stroke: '#e0e0e0' }}
          tickFormatter={(value) => `${value}%`}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend 
          wrapperStyle={{ paddingTop: '20px' }}
          iconType="rect"
        />
        <Bar 
          yAxisId="left"
          dataKey="revenue" 
          fill="#3498db" 
          name="Revenue"
          radius={[8, 8, 0, 0]}
        />
        <Bar 
          yAxisId="left"
          dataKey="profit" 
          fill="#27ae60" 
          name="Profit"
          radius={[8, 8, 0, 0]}
        />
        <Line 
          yAxisId="right"
          type="monotone" 
          dataKey="margin" 
          stroke="#e74c3c" 
          strokeWidth={3}
          name="Profit Margin %"
          dot={{ fill: '#e74c3c', r: 6 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default ProfitabilityChart;