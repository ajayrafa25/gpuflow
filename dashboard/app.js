const API_BASE = "";
let selectedJobId = null;

function getApiKey() {
  return localStorage.getItem("gpuflow_api_key") || "";
}

function promptApiKey() {
  const key = prompt("Enter your GPUFlow API key:", getApiKey());
  if (key !== null) {
    localStorage.setItem("gpuflow_api_key", key);
    document.getElementById("api-key-status").textContent = "Key set";
    refresh();
  }
}

async function apiFetch(path, options = {}) {
  const key = getApiKey();
  const res = await fetch(API_BASE + path, {
    ...options,
    headers: { "X-API-Key": key, "Content-Type": "application/json", ...(options.headers || {}) },
  });
  return res;
}

async function loadGPUs() {
  try {
    const res = await apiFetch("/api/v1/gpus");
    if (!res.ok) return;
    const gpus = await res.json();
    const container = document.getElementById("gpu-cards");
    if (!gpus.length) {
      container.innerHTML = '<p style="color:var(--muted)">No GPUs detected.</p>';
      return;
    }
    container.innerHTML = gpus.map(g => {
      const memPct = Math.round((g.used_memory_mb / g.total_memory_mb) * 100);
      const usedGiB = (g.used_memory_mb / 1024).toFixed(1);
      const totalGiB = (g.total_memory_mb / 1024).toFixed(1);
      return `
        <div class="gpu-card">
          <div class="gpu-index">GPU ${g.index}</div>
          <div class="gpu-name">${g.name}</div>
          <div class="progress-label">Utilization: ${g.utilization_pct}%</div>
          <div class="progress-bar"><div class="fill" style="width:${g.utilization_pct}%"></div></div>
          <div class="progress-label">Memory: ${usedGiB} / ${totalGiB} GiB (${memPct}%)</div>
          <div class="progress-bar"><div class="fill" style="width:${memPct}%"></div></div>
        </div>`;
    }).join("");
  } catch (e) {
    console.error("GPU load error", e);
  }
}

async function loadJobs() {
  try {
    const statusFilter = document.getElementById("status-filter").value;
    const qs = statusFilter ? `?status_filter=${statusFilter}` : "";
    const res = await apiFetch(`/api/v1/jobs${qs}`);
    if (!res.ok) return;
    const jobs = await res.json();
    const tbody = document.getElementById("jobs-body");
    const noJobs = document.getElementById("no-jobs");

    if (!jobs.length) {
      tbody.innerHTML = "";
      noJobs.style.display = "block";
      return;
    }
    noJobs.style.display = "none";

    tbody.innerHTML = jobs.map(job => {
      const created = job.created_at.replace("T", " ").slice(0, 19);
      const imgShort = job.docker_image.split("/").pop().slice(0, 28);
      const isTerminal = ["completed", "failed", "cancelled"].includes(job.status);
      const cancelBtn = isTerminal ? "" :
        `<button class="action-btn cancel" onclick="cancelJob('${job.id}', event)">Cancel</button>`;
      return `<tr onclick="showLogs('${job.id}')">
        <td style="font-family:monospace;font-size:12px">${job.id.slice(0, 12)}</td>
        <td>${esc(job.name)}</td>
        <td><span class="badge badge-${job.status}">${job.status}</span></td>
        <td>${job.requested_gpus}</td>
        <td style="color:var(--muted);font-size:12px">${esc(imgShort)}</td>
        <td style="color:var(--muted);font-size:12px">${created}</td>
        <td>
          <button class="action-btn" onclick="showLogs('${job.id}', event)">Logs</button>
          ${cancelBtn}
        </td>
      </tr>`;
    }).join("");
  } catch (e) {
    console.error("Jobs load error", e);
  }
}

async function showLogs(jobId, event) {
  if (event) event.stopPropagation();
  selectedJobId = jobId;
  document.getElementById("log-section").style.display = "block";
  document.getElementById("log-job-id").textContent = jobId.slice(0, 12);
  await refreshLogs();
}

async function refreshLogs() {
  if (!selectedJobId) return;
  try {
    const res = await apiFetch(`/api/v1/jobs/${selectedJobId}/logs`);
    if (res.ok) {
      const text = await res.text();
      const pre = document.getElementById("log-content");
      pre.textContent = text || "(no output yet)";
      pre.scrollTop = pre.scrollHeight;
    } else if (res.status === 404) {
      document.getElementById("log-content").textContent = "(no logs available yet)";
    }
  } catch (e) {
    console.error("Log load error", e);
  }
}

function closeLog() {
  selectedJobId = null;
  document.getElementById("log-section").style.display = "none";
}

async function cancelJob(jobId, event) {
  event.stopPropagation();
  if (!confirm(`Cancel job ${jobId.slice(0, 12)}?`)) return;
  await apiFetch(`/api/v1/jobs/${jobId}`, { method: "DELETE" });
  refresh();
}

function esc(str) {
  const d = document.createElement("div");
  d.appendChild(document.createTextNode(str));
  return d.innerHTML;
}

async function refresh() {
  await Promise.all([loadGPUs(), loadJobs()]);
  if (selectedJobId) await refreshLogs();
}

// Initial load + polling
if (!getApiKey()) promptApiKey();
refresh();
setInterval(refresh, 3000);
