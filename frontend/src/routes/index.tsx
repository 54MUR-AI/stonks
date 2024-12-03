import React from 'react';
import { Routes, Route } from 'react-router-dom';
import HealthDashboard from '../components/HealthDashboard';

const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/health" element={<HealthDashboard />} />
      {/* Add other routes here */}
    </Routes>
  );
};

export default AppRoutes;
