import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import {
  CheckCircle,
  Warning,
  Info,
  TrendingUp,
  Error,
} from '@mui/icons-material';

interface BusinessInsightsCardProps {
  category: string;
  insight: string;
  action: string;
  impact: string;
  type?: 'success' | 'warning' | 'info' | 'error';
}

const BusinessInsightsCard: React.FC<BusinessInsightsCardProps> = ({
  category,
  insight,
  action,
  impact,
  type = 'info',
}) => {
  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle />;
      case 'warning':
        return <Warning />;
      case 'error':
        return <Error />;
      default:
        return <Info />;
    }
  };

  const getColor = () => {
    switch (type) {
      case 'success':
        return '#27ae60';
      case 'warning':
        return '#f39c12';
      case 'error':
        return '#e74c3c';
      default:
        return '#3498db';
    }
  };

  const getBgColor = () => {
    switch (type) {
      case 'success':
        return 'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)';
      case 'warning':
        return 'linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%)';
      case 'error':
        return 'linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)';
      default:
        return 'linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)';
    }
  };

  return (
    <Card
      sx={{
        height: '100%',
        background: getBgColor(),
        borderLeft: `5px solid ${getColor()}`,
        borderRadius: 3,
        boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
        },
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 40,
              height: 40,
              borderRadius: 2,
              backgroundColor: getColor(),
              color: 'white',
              mr: 2,
            }}
          >
            {getIcon()}
          </Box>
          <Typography
            variant="h6"
            sx={{
              fontWeight: 600,
              color: getColor(),
            }}
          >
            {category}
          </Typography>
        </Box>

        <Typography
          variant="body1"
          sx={{
            mb: 2,
            color: '#2c3e50',
            fontWeight: 500,
          }}
        >
          {insight}
        </Typography>

        <Box sx={{ mb: 1 }}>
          <Typography
            variant="body2"
            sx={{
              color: '#7f8c8d',
              fontWeight: 600,
              mb: 0.5,
            }}
          >
            Recommended Action:
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: '#2c3e50',
            }}
          >
            {action}
          </Typography>
        </Box>

        <Box>
          <Typography
            variant="body2"
            sx={{
              color: '#7f8c8d',
              fontWeight: 600,
              mb: 0.5,
            }}
          >
            Expected Impact:
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: '#2c3e50',
              fontWeight: 500,
            }}
          >
            {impact}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default BusinessInsightsCard;