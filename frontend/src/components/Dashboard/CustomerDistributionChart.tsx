import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { Box } from '@mui/material';

interface CustomerDistributionChartProps {
  data?: any;
}

const CustomerDistributionChart: React.FC<CustomerDistributionChartProps> = ({ data }) => {
  const mockData = [
    { name: 'VIP', value: 15, percentage: 15 },
    { name: 'High CLV', value: 25, percentage: 25 },
    { name: 'Medium CLV', value: 35, percentage: 35 },
    { name: 'Low CLV', value: 25, percentage: 25 },
  ];

  const COLORS = {
    'VIP': '#9b59b6',
    'High CLV': '#27ae60',
    'Medium CLV': '#f39c12',
    'Low CLV': '#e74c3c',
  };

  const chartData = data?.byCLVTier 
    ? Object.entries(data.byCLVTier).map(([name, value]) => ({
        name,
        value,
        percentage: ((value as number) / (Object.values(data.byCLVTier).reduce((a: any, b: any) => a + b, 0) as number)) * 100
      }))
    : mockData;

  const CustomTooltip = ({ active, payload }: any) => {
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
          <p style={{ margin: 0, fontWeight: 600 }}>{payload[0].name}</p>
          <p style={{ margin: '4px 0', color: payload[0].fill }}>
            Count: {payload[0].value.toLocaleString()}
          </p>
          <p style={{ margin: '4px 0', color: '#7f8c8d' }}>
            Percentage: {payload[0].payload.percentage.toFixed(1)}%
          </p>
        </Box>
      );
    }
    return null;
  };

  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        fontWeight="600"
        fontSize="14"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={renderCustomizedLabel}
          outerRadius={120}
          fill="#8884d8"
          dataKey="value"
        >
          {chartData.map((entry: any, index: number) => (
            <Cell key={`cell-${index}`} fill={COLORS[entry.name as keyof typeof COLORS]} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend 
          verticalAlign="bottom" 
          height={36}
          formatter={(value, entry) => (
            <span style={{ color: entry.color }}>
              {value}: {entry.payload?.value?.toLocaleString() || 0}
            </span>
          )}
        />
      </PieChart>
    </ResponsiveContainer>
  );
};

export default CustomerDistributionChart;