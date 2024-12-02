import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Container,
  Tab,
  Tabs,
  Typography,
  Paper,
  CircularProgress,
  Alert,
} from '@mui/material';
import { StressTestDashboard } from '../components/StressTest';
import { PortfolioOverview } from '../components/Portfolio';
import { RiskMetrics } from '../components/Risk';
import { RiskPrediction } from '../components/Risk/RiskPrediction';
import { PortfolioRebalancing } from '../components/Portfolio/PortfolioRebalancing';
import { api } from '../services/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`portfolio-tabpanel-${index}`}
      aria-labelledby={`portfolio-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `portfolio-tab-${index}`,
    'aria-controls': `portfolio-tabpanel-${index}`,
  };
}

const Portfolio: React.FC = () => {
  const { portfolioId } = useParams<{ portfolioId: string }>();
  const [value, setValue] = useState(0);
  const [portfolio, setPortfolio] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPortfolio = async () => {
      try {
        const response = await api.get(`/portfolios/${portfolioId}`);
        setPortfolio(response.data);
      } catch (err) {
        setError('Failed to fetch portfolio data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolio();
  }, [portfolioId]);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container>
        <Alert severity="error" sx={{ mt: 3 }}>
          {error}
        </Alert>
      </Container>
    );
  }

  if (!portfolio) {
    return (
      <Container>
        <Alert severity="warning" sx={{ mt: 3 }}>
          Portfolio not found
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ mt: 4, mb: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          {portfolio.name}
        </Typography>
      </Box>

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={value}
            onChange={handleChange}
            aria-label="portfolio tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="Overview" {...a11yProps(0)} />
            <Tab label="Risk Metrics" {...a11yProps(1)} />
            <Tab label="Stress Testing" {...a11yProps(2)} />
            <Tab label="Risk Prediction" {...a11yProps(3)} />
            <Tab label="Rebalancing" {...a11yProps(4)} />
          </Tabs>
        </Box>

        <TabPanel value={value} index={0}>
          <PortfolioOverview portfolio={portfolio} />
        </TabPanel>

        <TabPanel value={value} index={1}>
          <RiskMetrics portfolioId={parseInt(portfolioId)} />
        </TabPanel>

        <TabPanel value={value} index={2}>
          <StressTestDashboard portfolioId={parseInt(portfolioId)} />
        </TabPanel>

        <TabPanel value={value} index={3}>
          <RiskPrediction portfolioId={parseInt(portfolioId)} />
        </TabPanel>

        <TabPanel value={value} index={4}>
          <PortfolioRebalancing portfolioId={parseInt(portfolioId)} />
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default Portfolio;
