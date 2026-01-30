// When running in Docker, use relative URLs (nginx proxies to backend)
// When running locally for development, fall back to localhost:8000
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

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
  created_at?: string;
  uptime?: string;
  image?: string;
}

export interface DeploymentRequest {
  launchable_id: string;
  ngc_api_key?: string;
  hf_api_key?: string;
}

export interface DeploymentJob {
  id: string;
  launchable_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'dry_run';
  commands: string[];
  current_step: number;
  total_steps: number;
  logs: string[];
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export interface InstructionsResponse {
  launchable_id: string;
  commands: string[];
  raw_html?: string;
}

export interface AutoDeployRequest {
  launchable_id: string;
  ngc_api_key?: string;
  hf_api_key?: string;
  dry_run?: boolean;
}

// Fetch system stats from nvidia-smi
export async function fetchSystemStats(): Promise<SystemStats> {
  const url = `${API_BASE_URL}/api/system/stats`;
  console.log('[API] Fetching system stats from:', url);
  
  const response = await fetch(url);
  console.log('[API] System stats response:', response.status, response.headers.get('content-type'));
  
  if (!response.ok) {
    const text = await response.text();
    console.error('[API] System stats error:', text.substring(0, 200));
    throw new Error(`Failed to fetch system stats: ${response.statusText}`);
  }
  
  const data = await response.json();
  console.log('[API] System stats data:', data);
  return data;
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

// Health check - verifies backend is responding with valid JSON
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) return false;
    
    // Verify it's actually JSON from the backend, not HTML from Vite/nginx fallback
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      return false;
    }
    
    // Parse to ensure valid JSON response
    const data = await response.json();
    return data && data.status === 'healthy';
  } catch {
    return false;
  }
}

// Get container logs
export async function getContainerLogs(launchableId: string, tail = 100): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/api/deployments/${launchableId}/logs?tail=${tail}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch logs: ${response.statusText}`);
  }
  const data = await response.json();
  return data.logs;
}

// WebSocket URL for log streaming
export function getLogsWebSocketUrl(launchableId: string): string {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsHost = API_BASE_URL.replace(/^https?:\/\//, '') || window.location.host;
  return `${wsProtocol}//${wsHost}/ws/logs/${launchableId}`;
}

// ============ Auto-Deploy API ============

// Fetch installation instructions from NVIDIA
export async function fetchInstructions(launchableId: string): Promise<InstructionsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/launchables/${launchableId}/instructions`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to fetch instructions for ${launchableId}`);
  }
  return response.json();
}

// Start automated deployment
export async function startAutoDeploy(request: AutoDeployRequest): Promise<DeploymentJob> {
  const response = await fetch(`${API_BASE_URL}/api/launchables/${request.launchable_id}/auto-deploy`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to start auto-deploy for ${request.launchable_id}`);
  }
  return response.json();
}

// Get deployment job status
export async function getDeploymentJob(jobId: string): Promise<DeploymentJob> {
  const response = await fetch(`${API_BASE_URL}/api/deployments/jobs/${jobId}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to get job ${jobId}`);
  }
  return response.json();
}

// List all deployment jobs
export async function listDeploymentJobs(): Promise<DeploymentJob[]> {
  const response = await fetch(`${API_BASE_URL}/api/deployments/jobs`);
  if (!response.ok) {
    throw new Error(`Failed to list deployment jobs: ${response.statusText}`);
  }
  return response.json();
}

// WebSocket URL for job log streaming
export function getJobLogsWebSocketUrl(jobId: string): string {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsHost = API_BASE_URL.replace(/^https?:\/\//, '') || window.location.host;
  return `${wsProtocol}//${wsHost}/ws/jobs/${jobId}`;
}

// ============ Terminal API ============

export interface TerminalRequest {
  working_directory?: string;
  command?: string;
}

export interface TerminalResponse {
  success: boolean;
  terminal: string;
  message: string;
}

// Open a terminal on the DGX Spark host
export async function openTerminal(request?: TerminalRequest): Promise<TerminalResponse> {
  const response = await fetch(`${API_BASE_URL}/api/terminal/open`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request || {}),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to open terminal');
  }
  return response.json();
}
