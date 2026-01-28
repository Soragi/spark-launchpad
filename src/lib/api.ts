const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface SystemStats {
  memory_used: number;
  memory_total: number;
  gpu_utilization: number;
  gpu_memory_used: number;
  gpu_memory_total: number;
  gpu_name: string;
  gpu_temperature: number;
  cuda_version: string;
  driver_version: string;
}

export interface DeploymentStatus {
  id: string;
  name: string;
  status: string;
  container_id?: string;
  ports?: Record<string, string>;
  error?: string;
}

export interface DeploymentRequest {
  launchable_id: string;
  ngc_api_key?: string;
  hf_api_key?: string;
}

// Fetch system stats from nvidia-smi
export async function fetchSystemStats(): Promise<SystemStats> {
  const response = await fetch(`${API_BASE_URL}/api/system/stats`);
  if (!response.ok) {
    throw new Error(`Failed to fetch system stats: ${response.statusText}`);
  }
  return response.json();
}

// List all deployments
export async function listDeployments(): Promise<DeploymentStatus[]> {
  const response = await fetch(`${API_BASE_URL}/api/deployments`);
  if (!response.ok) {
    throw new Error(`Failed to list deployments: ${response.statusText}`);
  }
  return response.json();
}

// Deploy a launchable
export async function deployLaunchable(
  launchableId: string,
  ngcApiKey?: string,
  hfApiKey?: string
): Promise<DeploymentStatus> {
  const response = await fetch(`${API_BASE_URL}/api/deployments/${launchableId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      launchable_id: launchableId,
      ngc_api_key: ngcApiKey,
      hf_api_key: hfApiKey,
    }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to deploy ${launchableId}`);
  }
  return response.json();
}

// Stop a deployment
export async function stopDeployment(launchableId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/deployments/${launchableId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to stop ${launchableId}`);
  }
}

// Manage JupyterLab service
export async function manageJupyterLab(action: 'start' | 'stop' | 'restart'): Promise<DeploymentStatus> {
  const response = await fetch(`${API_BASE_URL}/api/services/jupyterlab`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ action }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to ${action} JupyterLab`);
  }
  return response.json();
}

// Health check
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return response.ok;
  } catch {
    return false;
  }
}
