import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Divider,
  IconButton,
  Avatar,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Segment as SegmentIcon,
  TrendingDown as ChurnIcon,
  Science as ABTestIcon,
  Analytics as CohortIcon,
  Recommend as RecommendIcon,
  Lightbulb as InsightsIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material';

interface SidebarProps {
  open: boolean;
  onToggle: () => void;
}

const menuItems = [
  {
    title: 'Executive Dashboard',
    path: '/dashboard',
    icon: <DashboardIcon />,
    color: '#667eea',
  },
  {
    title: 'Advanced Segmentation',
    path: '/segmentation',
    icon: <SegmentIcon />,
    color: '#764ba2',
  },
  {
    title: 'Churn & CLV Analysis',
    path: '/churn-clv',
    icon: <ChurnIcon />,
    color: '#e74c3c',
  },
  {
    title: 'A/B Testing Suite',
    path: '/ab-testing',
    icon: <ABTestIcon />,
    color: '#3498db',
  },
  {
    title: 'Cohort Analysis',
    path: '/cohort-analysis',
    icon: <CohortIcon />,
    color: '#27ae60',
  },
  {
    title: 'Recommendation Engine',
    path: '/recommendations',
    icon: <RecommendIcon />,
    color: '#f39c12',
  },
  {
    title: 'Business Insights',
    path: '/insights',
    icon: <InsightsIcon />,
    color: '#9b59b6',
  },
];

const Sidebar: React.FC<SidebarProps> = ({ open, onToggle }) => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: open ? 280 : 64,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: open ? 280 : 64,
          boxSizing: 'border-box',
          background: 'linear-gradient(180deg, #2c3e50 0%, #34495e 100%)',
          color: 'white',
          transition: 'width 0.3s ease',
          overflowX: 'hidden',
        },
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: open ? 'space-between' : 'center',
          p: 2,
          minHeight: 64,
        }}
      >
        {open && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Avatar
              sx={{
                width: 40,
                height: 40,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              }}
            >
              CA
            </Avatar>
            <Box>
              <Typography variant="h6" sx={{ color: 'white', fontWeight: 700 }}>
                Analytics
              </Typography>
              <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                Platform v2.0
              </Typography>
            </Box>
          </Box>
        )}
        <IconButton onClick={onToggle} sx={{ color: 'white' }}>
          {open ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </IconButton>
      </Box>

      <Divider sx={{ backgroundColor: 'rgba(255,255,255,0.1)' }} />

      <List sx={{ mt: 2 }}>
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <ListItem key={item.path} disablePadding sx={{ mb: 1 }}>
              <ListItemButton
                onClick={() => navigate(item.path)}
                sx={{
                  mx: 1,
                  borderRadius: 2,
                  backgroundColor: isActive ? 'rgba(255,255,255,0.1)' : 'transparent',
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.15)',
                  },
                  position: 'relative',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    left: 0,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    width: 4,
                    height: isActive ? '70%' : 0,
                    backgroundColor: item.color,
                    borderRadius: '0 4px 4px 0',
                    transition: 'height 0.3s ease',
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    color: isActive ? item.color : 'rgba(255,255,255,0.7)',
                    minWidth: open ? 40 : 'auto',
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                {open && (
                  <ListItemText
                    primary={item.title}
                    primaryTypographyProps={{
                      fontSize: '0.9rem',
                      fontWeight: isActive ? 600 : 400,
                      color: isActive ? 'white' : 'rgba(255,255,255,0.9)',
                    }}
                  />
                )}
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      <Box sx={{ flexGrow: 1 }} />

      {open && (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)' }}>
            © 2024 Customer Analytics
          </Typography>
        </Box>
      )}
    </Drawer>
  );
};

export default Sidebar;