import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Card,
  CardContent,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  TrendingUp,
  Warning,
  CheckCircle,
  Error,
} from '@mui/icons-material';

interface InsightItem {
  category: string;
  insight: string;
  action: string;
  impact: string;
  priority: 'high' | 'medium' | 'low';
  type: 'success' | 'warning' | 'error' | 'info';
}

const BusinessInsights: React.FC = () => {
  const insights: InsightItem[] = [
    {
      category: 'Customer Value',
      insight: 'VIP customers generate 65% of total revenue despite being only 5% of customer base',
      action: 'Implement VIP-specific retention programs and exclusive benefits',
      impact: 'Potential 15% revenue increase through improved VIP retention',
      priority: 'high',
      type: 'success',
    },
    {
      category: 'Retention',
      insight: 'Churn rate increased by 5% in Q4 compared to Q3',
      action: 'Launch targeted win-back campaigns for high-value churned customers',
      impact: 'Recover $2.5M in potential lost revenue',
      priority: 'high',
      type: 'error',
    },
    {
      category: 'Growth',
      insight: 'New customer acquisition cost decreased by 20% through referral programs',
      action: 'Scale successful referral programs and incentivize more customer advocacy',
      impact: '$4M additional revenue with improved CAC',
      priority: 'medium',
      type: 'success',
    },
    {
      category: 'Product Performance',
      insight: 'Electronics category shows 40% higher profit margins than average',
      action: 'Increase marketing spend and inventory for electronics',
      impact: 'Projected 25% increase in profit from category expansion',
      priority: 'medium',
      type: 'success',
    },
    {
      category: 'Seasonal Trends',
      insight: 'Q4 shows 35% higher sales but 50% higher support costs',
      action: 'Implement automated support solutions and seasonal staff training',
      impact: 'Reduce support costs by 30% while maintaining satisfaction',
      priority: 'medium',
      type: 'warning',
    },
    {
      category: 'Customer Behavior',
      insight: 'Mobile purchases increased 45% YoY but have 20% lower AOV',
      action: 'Optimize mobile checkout and implement mobile-specific upsells',
      impact: 'Increase mobile AOV by 15%',
      priority: 'low',
      type: 'info',
    },
  ];

  const getIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle />;
      case 'warning':
        return <Warning />;
      case 'error':
        return <Error />;
      default:
        return <TrendingUp />;
    }
  };

  const getColor = (type: string) => {
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

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      default:
        return 'default';
    }
  };

  const groupedInsights = insights.reduce((acc, insight) => {
    if (!acc[insight.category]) {
      acc[insight.category] = [];
    }
    acc[insight.category].push(insight);
    return acc;
  }, {} as Record<string, InsightItem[]>);

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 4, fontWeight: 700 }}>
        Business Insights & Recommendations
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#d4edda', borderLeft: '4px solid #27ae60' }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: '#155724' }}>
                Total Insights
              </Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: '#155724' }}>
                {insights.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#f8d7da', borderLeft: '4px solid #e74c3c' }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: '#721c24' }}>
                High Priority
              </Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: '#721c24' }}>
                {insights.filter(i => i.priority === 'high').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#fff3cd', borderLeft: '4px solid #f39c12' }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: '#856404' }}>
                Medium Priority
              </Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: '#856404' }}>
                {insights.filter(i => i.priority === 'medium').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ backgroundColor: '#e3f2fd', borderLeft: '4px solid #3498db' }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: '#0c5460' }}>
                Revenue Impact
              </Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: '#0c5460' }}>
                $8.5M
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Paper sx={{ p: 3, borderRadius: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
          Insights by Category
        </Typography>
        
        {Object.entries(groupedInsights).map(([category, categoryInsights]) => (
          <Accordion key={category} sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                <Typography sx={{ fontWeight: 600, flexGrow: 1 }}>
                  {category}
                </Typography>
                <Chip 
                  label={`${categoryInsights.length} insights`} 
                  size="small" 
                  sx={{ mr: 2 }}
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {categoryInsights.map((insight, index) => (
                  <Grid item xs={12} key={index}>
                    <Card 
                      sx={{ 
                        borderLeft: `4px solid ${getColor(insight.type)}`,
                        backgroundColor: '#f8f9fa'
                      }}
                    >
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Box
                            sx={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              width: 36,
                              height: 36,
                              borderRadius: 1,
                              backgroundColor: getColor(insight.type),
                              color: 'white',
                              mr: 2,
                            }}
                          >
                            {getIcon(insight.type)}
                          </Box>
                          <Typography variant="subtitle1" sx={{ fontWeight: 600, flexGrow: 1 }}>
                            {insight.insight}
                          </Typography>
                          <Chip 
                            label={insight.priority} 
                            size="small" 
                            color={getPriorityColor(insight.priority) as any}
                          />
                        </Box>
                        
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                              Recommended Action:
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {insight.action}
                            </Typography>
                          </Grid>
                          <Grid item xs={12} md={6}>
                            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                              Expected Impact:
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {insight.impact}
                            </Typography>
                          </Grid>
                        </Grid>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))}
      </Paper>
    </Box>
  );
};

export default BusinessInsights;