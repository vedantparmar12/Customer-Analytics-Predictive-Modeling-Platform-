import React, { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, Container } from '@mui/material';
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import ExecutiveDashboard from './pages/ExecutiveDashboard';
import AdvancedSegmentation from './pages/AdvancedSegmentation';
import ChurnCLVAnalysis from './pages/ChurnCLVAnalysis';
import ABTestingSuite from './pages/ABTestingSuite';
import CohortAnalysis from './pages/CohortAnalysis';
import RecommendationEngine from './pages/RecommendationEngine';
import BusinessInsights from './pages/BusinessInsights';

const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar open={sidebarOpen} onToggle={toggleSidebar} />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          marginLeft: sidebarOpen ? '280px' : '64px',
          transition: 'margin 0.3s ease',
          width: sidebarOpen ? 'calc(100% - 280px)' : 'calc(100% - 64px)',
        }}
      >
        <Header onMenuClick={toggleSidebar} />
        <Container 
          maxWidth={false} 
          sx={{ 
            mt: 2, 
            mb: 4, 
            px: { xs: 2, sm: 3, md: 4 },
            flexGrow: 1,
          }}
        >
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<ExecutiveDashboard />} />
            <Route path="/segmentation" element={<AdvancedSegmentation />} />
            <Route path="/churn-clv" element={<ChurnCLVAnalysis />} />
            <Route path="/ab-testing" element={<ABTestingSuite />} />
            <Route path="/cohort-analysis" element={<CohortAnalysis />} />
            <Route path="/recommendations" element={<RecommendationEngine />} />
            <Route path="/insights" element={<BusinessInsights />} />
          </Routes>
        </Container>
      </Box>
    </Box>
  );
};

export default App;