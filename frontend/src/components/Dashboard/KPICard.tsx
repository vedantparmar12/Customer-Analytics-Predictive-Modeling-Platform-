import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';

interface KPICardProps {
  title: string;
  value: string;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon?: React.ReactNode;
  color?: string;
}

const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  change,
  trend = 'neutral',
  icon,
  color = '#667eea',
}) => {
  const getTrendIcon = () => {
    if (trend === 'up') return <TrendingUp sx={{ fontSize: 16 }} />;
    if (trend === 'down') return <TrendingDown sx={{ fontSize: 16 }} />;
    return null;
  };

  const getTrendColor = () => {
    if (trend === 'up') return '#27ae60';
    if (trend === 'down') return '#e74c3c';
    return '#7f8c8d';
  };

  return (
    <Card
      sx={{
        height: '100%',
        position: 'relative',
        overflow: 'visible',
        background: 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
        borderRadius: 3,
        boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
        },
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: -10,
          left: 20,
          width: 40,
          height: 40,
          borderRadius: 2,
          background: `linear-gradient(135deg, ${color} 0%, ${color}dd 100%)`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          boxShadow: '0 4px 8px rgba(0,0,0,0.15)',
        }}
      >
        {icon}
      </Box>

      <CardContent sx={{ pt: 4 }}>
        <Typography
          variant="body2"
          sx={{
            color: '#7f8c8d',
            fontWeight: 600,
            textTransform: 'uppercase',
            fontSize: '0.75rem',
            letterSpacing: '0.5px',
            mb: 1,
          }}
        >
          {title}
        </Typography>

        <Typography
          variant="h4"
          sx={{
            color: '#2c3e50',
            fontWeight: 700,
            mb: 1,
          }}
        >
          {value}
        </Typography>

        {change && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {getTrendIcon()}
            <Typography
              variant="body2"
              sx={{
                color: getTrendColor(),
                fontWeight: 500,
                fontSize: '0.875rem',
              }}
            >
              {change}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default KPICard;