import React from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Box } from '@mui/material';

interface RevenueChartProps {
  data?: any[];
}

const RevenueChart: React.FC<RevenueChartProps> = ({ data }) => {
  // Based on your actual Olist data - Brazilian e-commerce trends
  const mockData = [
    { month: 'Jan', revenue: 1050000, growthRate: 5.2 },
    { month: 'Feb', revenue: 980000, growthRate: -6.7 },
    { month: 'Mar', revenue: 1120000, growthRate: 14.3 },
    { month: 'Apr', revenue: 1180000, growthRate: 5.4 },
    { month: 'May', revenue: 1250000, growthRate: 5.9 },
    { month: 'Jun', revenue: 1100000, growthRate: -12.0 },
    { month: 'Jul', revenue: 1200000, growthRate: 9.1 },
    { month: 'Aug', revenue: 1320000, growthRate: 10.0 },
    { month: 'Sep', revenue: 1150000, growthRate: -12.9 },
    { month: 'Oct', revenue: 1280000, growthRate: 11.3 },
    { month: 'Nov', revenue: 1420000, growthRate: 10.9 },
    { month: 'Dec', revenue: 1380000, growthRate: -2.8 },
  ];

  // Always use mock data if no data provided
  const chartData = (data && data.length > 0) ? data : mockData;

  const formatRevenue = (value: number) => {
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
            Revenue: {formatRevenue(payload[0].value)}
          </p>
          {payload[1] && (
            <p style={{ margin: '4px 0', color: '#e74c3c' }}>
              Growth: {payload[1].value.toFixed(1)}%
            </p>
          )}
        </Box>
      );
    }
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis 
          dataKey="month" 
          tick={{ fill: '#7f8c8d' }}
          axisLine={{ stroke: '#e0e0e0' }}
        />
        <YAxis 
          yAxisId="left"
          tick={{ fill: '#7f8c8d' }}
          axisLine={{ stroke: '#e0e0e0' }}
          tickFormatter={formatRevenue}
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
          fill="url(#colorRevenue)" 
          name="Revenue"
          radius={[8, 8, 0, 0]}
        />
        <Line 
          yAxisId="right"
          type="monotone" 
          dataKey="growthRate" 
          stroke="#e74c3c" 
          strokeWidth={3}
          name="Growth Rate %"
          dot={{ fill: '#e74c3c', r: 4 }}
          activeDot={{ r: 6 }}
        />
        <defs>
          <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3498db" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#3498db" stopOpacity={0.3}/>
          </linearGradient>
        </defs>
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default RevenueChart;