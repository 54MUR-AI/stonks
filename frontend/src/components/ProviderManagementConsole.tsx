import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  Tooltip,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
} from '@mui/icons-material';

interface Provider {
  id: string;
  name: string;
  type: string;
  priority: 'PRIMARY' | 'SECONDARY' | 'FALLBACK';
  status: 'ACTIVE' | 'INACTIVE';
  health: {
    status: string;
    metrics: Record<string, any>;
  };
  config: Record<string, any>;
}

interface ProviderFormData {
  name: string;
  type: string;
  priority: string;
  config: Record<string, string>;
}

const ProviderManagementConsole: React.FC = () => {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(null);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [formData, setFormData] = useState<ProviderFormData>({
    name: '',
    type: '',
    priority: 'SECONDARY',
    config: {},
  });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchProviders();
    const interval = setInterval(fetchProviders, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchProviders = async () => {
    try {
      const response = await fetch('/api/providers');
      const data = await response.json();
      setProviders(data);
    } catch (err) {
      setError('Failed to fetch providers');
    }
  };

  const handleAddProvider = async () => {
    try {
      const response = await fetch('/api/providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!response.ok) throw new Error('Failed to add provider');
      
      await fetchProviders();
      setIsAddDialogOpen(false);
      resetForm();
    } catch (err) {
      setError('Failed to add provider');
    }
  };

  const handleEditProvider = async () => {
    if (!selectedProvider) return;
    
    try {
      const response = await fetch(`/api/providers/${selectedProvider.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!response.ok) throw new Error('Failed to update provider');
      
      await fetchProviders();
      setIsEditDialogOpen(false);
      resetForm();
    } catch (err) {
      setError('Failed to update provider');
    }
  };

  const handleDeleteProvider = async (providerId: string) => {
    try {
      const response = await fetch(`/api/providers/${providerId}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete provider');
      
      await fetchProviders();
    } catch (err) {
      setError('Failed to delete provider');
    }
  };

  const handleStartProvider = async (providerId: string) => {
    try {
      const response = await fetch(`/api/providers/${providerId}/start`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to start provider');
      
      await fetchProviders();
    } catch (err) {
      setError('Failed to start provider');
    }
  };

  const handleStopProvider = async (providerId: string) => {
    try {
      const response = await fetch(`/api/providers/${providerId}/stop`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to stop provider');
      
      await fetchProviders();
    } catch (err) {
      setError('Failed to stop provider');
    }
  };

  const resetForm = () => {
    setFormData({
      name: '',
      type: '',
      priority: 'SECONDARY',
      config: {},
    });
    setSelectedProvider(null);
  };

  const getHealthColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy': return 'success.main';
      case 'degraded': return 'warning.main';
      case 'unhealthy': return 'error.main';
      default: return 'grey.500';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Provider Management Console</Typography>
        <Box>
          <IconButton onClick={fetchProviders} sx={{ mr: 1 }}>
            <RefreshIcon />
          </IconButton>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setIsAddDialogOpen(true)}
          >
            Add Provider
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Priority</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Health</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {providers.map((provider) => (
              <TableRow key={provider.id}>
                <TableCell>{provider.name}</TableCell>
                <TableCell>{provider.type}</TableCell>
                <TableCell>{provider.priority}</TableCell>
                <TableCell>
                  <Typography color={provider.status === 'ACTIVE' ? 'success.main' : 'text.secondary'}>
                    {provider.status}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography color={getHealthColor(provider.health.status)}>
                    {provider.health.status}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Box>
                    {provider.status === 'INACTIVE' ? (
                      <Tooltip title="Start Provider">
                        <IconButton
                          onClick={() => handleStartProvider(provider.id)}
                          color="success"
                          size="small"
                        >
                          <StartIcon />
                        </IconButton>
                      </Tooltip>
                    ) : (
                      <Tooltip title="Stop Provider">
                        <IconButton
                          onClick={() => handleStopProvider(provider.id)}
                          color="error"
                          size="small"
                        >
                          <StopIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Tooltip title="Edit Provider">
                      <IconButton
                        onClick={() => {
                          setSelectedProvider(provider);
                          setFormData({
                            name: provider.name,
                            type: provider.type,
                            priority: provider.priority,
                            config: provider.config,
                          });
                          setIsEditDialogOpen(true);
                        }}
                        size="small"
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete Provider">
                      <IconButton
                        onClick={() => handleDeleteProvider(provider.id)}
                        color="error"
                        size="small"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Add Provider Dialog */}
      <Dialog open={isAddDialogOpen} onClose={() => setIsAddDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Add Provider</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Provider Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Provider Type</InputLabel>
                <Select
                  value={formData.type}
                  label="Provider Type"
                  onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                >
                  <MenuItem value="ALPHA_VANTAGE">Alpha Vantage</MenuItem>
                  <MenuItem value="YAHOO_FINANCE">Yahoo Finance</MenuItem>
                  <MenuItem value="MOCK">Mock Provider</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Priority</InputLabel>
                <Select
                  value={formData.priority}
                  label="Priority"
                  onChange={(e) => setFormData({ ...formData, priority: e.target.value })}
                >
                  <MenuItem value="PRIMARY">Primary</MenuItem>
                  <MenuItem value="SECONDARY">Secondary</MenuItem>
                  <MenuItem value="FALLBACK">Fallback</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsAddDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddProvider} variant="contained">Add</Button>
        </DialogActions>
      </Dialog>

      {/* Edit Provider Dialog */}
      <Dialog open={isEditDialogOpen} onClose={() => setIsEditDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Edit Provider</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Provider Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Priority</InputLabel>
                <Select
                  value={formData.priority}
                  label="Priority"
                  onChange={(e) => setFormData({ ...formData, priority: e.target.value })}
                >
                  <MenuItem value="PRIMARY">Primary</MenuItem>
                  <MenuItem value="SECONDARY">Secondary</MenuItem>
                  <MenuItem value="FALLBACK">Fallback</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsEditDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleEditProvider} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProviderManagementConsole;
