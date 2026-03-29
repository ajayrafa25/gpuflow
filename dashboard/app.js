const API_BASE = "";
let selectedJobId = null;
let latestGPUs = [];
let latestJobs = [];

// Chart.js instances and history
const gpuCharts = {};       // { gpuIndex: Chart }
const gpuHistory = {};      // { gpuIndex: number[] }
let donutChart = null;

// ─── Auth ────────────────────────────────────────────────────────────────────

function getApiKey() {
  return localStorage.getItem("gpuflow_api_key") || "";
}

function promptApiKey() {
  const key = prompt("Enter your GPUFlow API key:", getApiKey());
  if (key !== null) {
    localStorage.setItem("gpuflow_api_key", key);
    document.getElementById("api-key-status").textContent = "Key set ✓";
    refresh();
  }
}

async function apiFetch(path, options = {}) {
  const key = getApiKey();
  return fetch(API_BASE + path, {
    ...options,
    headers: {
      "X-API-Key": key,
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
}

function esc(str) {
  const d = document.createElement("div");
  d.appendChild(document.createTextNode(String(str)));
  return d.innerHTML;
}

// ─── Color helpers ────────────────────────────────────────────────────────────

function userColor(name, alpha = 1) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  const h = Math.abs(hash) % 360;
  return `hsla(${h}, 65%, 55%, ${alpha})`;
}

// ─── GPU Section ─────────────────────────────────────────────────────────────

function buildGaugeSVG(pct, color) {
  const r = 38, cx = 50, cy = 50;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - Math.min(pct, 100) / 100);
  return `<svg viewBox="0 0 100 100" class="gauge-svg">
    <circle cx="${cx}" cy="${cy}" r="${r}" class="gauge-bg"/>
    <circle cx="${cx}" cy="${cy}" r="${r}"
      class="gauge-fill"
      stroke="${color}"
      stroke-dasharray="${circ}"
      stroke-dashoffset="${offset}"
      transform="rotate(-90 ${cx} ${cy})"/>
    <text x="${cx}" y="${cy + 5}" class="gauge-text">${pct}%</text>
  </svg>`;
}

async function loadGPUs() {
  try {
    const res = await apiFetch("/api/v1/gpus");
    if (!res.ok) return;
    latestGPUs = await res.json();
    const container = document.getElementById("gpu-cards");

    if (!latestGPUs.length) {
      container.innerHTML = '<p class="muted">No GPUs detected.</p>';
      return;
    }

    latestGPUs.forEach(g => {
      // Update history
      if (!gpuHistory[g.index]) gpuHistory[g.index] = [];
      gpuHistory[g.index].push(g.utilization_pct);
      if (gpuHistory[g.index].length > 30) gpuHistory[g.index].shift();

      const memPct = Math.round((g.used_memory_mb / g.total_memory_mb) * 100);
      const usedGiB = (g.used_memory_mb / 1024).toFixed(1);
      const totalGiB = (g.total_memory_mb / 1024).toFixed(1);
      const utilColor = g.utilization_pct > 80 ? "#f87171" : g.utilization_pct > 40 ? "#fbbf24" : "#34d399";
      const memColor  = memPct > 85 ? "#f87171" : memPct > 60 ? "#fbbf24" : "#818cf8";
      const cardId = `gpu-card-${g.index}`;
      const existing = document.getElementById(cardId);

      if (!existing) {
        // First render: create card
        const card = document.createElement("div");
        card.className = "gpu-card";
        card.id = cardId;
        card.innerHTML = `
          <div class="gpu-header">
            <span class="gpu-index">GPU ${g.index}</span>
            <span class="gpu-name">${esc(g.name)}</span>
            <span class="gpu-avail ${g.is_available ? 'avail' : 'busy'}">${g.is_available ? "Free" : "Busy"}</span>
          </div>
          <div class="gauge-row">
            <div class="gauge-wrap">
              <div id="gauge-util-${g.index}">${buildGaugeSVG(g.utilization_pct, utilColor)}</div>
              <div class="gauge-label">Utilization</div>
            </div>
            <div class="gauge-wrap">
              <div id="gauge-mem-${g.index}">${buildGaugeSVG(memPct, memColor)}</div>
              <div class="gauge-label">${usedGiB} / ${totalGiB} GiB</div>
            </div>
          </div>
          <div class="sparkline-wrap">
            <canvas id="spark-${g.index}" height="50"></canvas>
          </div>`;
        container.appendChild(card);

        // Init sparkline chart
        const ctx = document.getElementById(`spark-${g.index}`).getContext("2d");
        gpuCharts[g.index] = new Chart(ctx, {
          type: "line",
          data: {
            labels: Array(30).fill(""),
            datasets: [{
              data: [...gpuHistory[g.index]],
              borderColor: utilColor,
              backgroundColor: utilColor.replace(")", ", 0.15)").replace("hsl", "hsla"),
              borderWidth: 1.5,
              fill: true,
              tension: 0.4,
              pointRadius: 0,
            }],
          },
          options: {
            animation: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
              x: { display: false },
              y: { display: false, min: 0, max: 100 },
            },
          },
        });
      } else {
        // Update in-place
        document.getElementById(`gauge-util-${g.index}`).innerHTML = buildGaugeSVG(g.utilization_pct, utilColor);
        document.getElementById(`gauge-mem-${g.index}`).innerHTML = buildGaugeSVG(memPct, memColor);
        existing.querySelector(".gpu-avail").textContent = g.is_available ? "Free" : "Busy";
        existing.querySelector(".gpu-avail").className = `gpu-avail ${g.is_available ? "avail" : "busy"}`;
        existing.querySelectorAll(".gauge-label")[1].textContent = `${usedGiB} / ${totalGiB} GiB`;

        const chart = gpuCharts[g.index];
        if (chart) {
          chart.data.datasets[0].data = [...gpuHistory[g.index]];
          chart.update("none");
        }
      }
    });
  } catch (e) {
    console.error("GPU load error", e);
  }
}

// ─── Resource Panel ──────────────────────────────────────────────────────────

function renderResourcePanel(jobs, gpus) {
  if (!gpus.length) return;

  const runningJobs = jobs.filter(j => j.status === "running" && j.assigned_gpus.length);

  // Build per-user memory usage
  const userMemory = {};
  let allocatedMb = 0;
  for (const job of runningJobs) {
    const user = job.submitted_by || "anonymous";
    const memPerGpu = gpus.length ? gpus[0].total_memory_mb : 0;
    const jobMem = job.assigned_gpus.length * memPerGpu;
    userMemory[user] = (userMemory[user] || 0) + jobMem;
    allocatedMb += jobMem;
  }
  const totalMb = gpus.reduce((s, g) => s + g.total_memory_mb, 0);
  const freeMb = Math.max(0, totalMb - allocatedMb);

  // Donut chart
  const users = Object.keys(userMemory);
  const labels = [...users.map(u => `${u} (${(userMemory[u] / 1024).toFixed(1)} GiB)`), `Free (${(freeMb / 1024).toFixed(1)} GiB)`];
  const data   = [...users.map(u => userMemory[u]), freeMb];
  const colors = [...users.map(u => userColor(u)), "rgba(255,255,255,0.08)"];

  if (donutChart) {
    donutChart.data.labels = labels;
    donutChart.data.datasets[0].data = data;
    donutChart.data.datasets[0].backgroundColor = colors;
    donutChart.update("none");
  } else {
    const ctx = document.getElementById("user-donut-chart");
    if (ctx) {
      donutChart = new Chart(ctx, {
        type: "doughnut",
        data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 2, borderColor: "#1e1e2e" }] },
        options: {
          cutout: "65%",
          animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => ` ${ctx.label}`,
              },
            },
          },
        },
      });
    }
  }

  // Legend
  const legend = document.getElementById("donut-legend");
  if (legend) {
    legend.innerHTML = labels.map((l, i) =>
      `<div class="legend-item"><span class="legend-dot" style="background:${colors[i]}"></span>${esc(l)}</div>`
    ).join("");
  }

  // GPU allocation bar
  const bar = document.getElementById("gpu-allocation-bar");
  if (!bar) return;

  // Map gpu index → running job
  const gpuOwner = {};
  for (const job of runningJobs) {
    for (const idx of job.assigned_gpus) {
      gpuOwner[idx] = job.submitted_by || "anonymous";
    }
  }

  bar.innerHTML = gpus.map(g => {
    const owner = gpuOwner[g.index];
    const bg = owner ? userColor(owner, 0.25) : "rgba(255,255,255,0.05)";
    const border = owner ? userColor(owner) : "rgba(255,255,255,0.1)";
    const label = owner ? `${owner}` : "free";
    return `<div class="gpu-bar-row">
      <span class="gpu-bar-label">GPU ${g.index}</span>
      <div class="gpu-bar-seg" style="background:${bg};border-color:${border}">
        <span class="gpu-bar-user" style="color:${owner ? userColor(owner) : "var(--muted)"}">${esc(label)}</span>
      </div>
    </div>`;
  }).join("");
}

// ─── Jobs ─────────────────────────────────────────────────────────────────────

async function loadJobs() {
  try {
    const statusFilter = document.getElementById("status-filter").value;
    const qs = statusFilter ? `?status_filter=${statusFilter}` : "";
    const res = await apiFetch(`/api/v1/jobs${qs}`);
    if (!res.ok) return;
    latestJobs = await res.json();

    const tbody = document.getElementById("jobs-body");
    const noJobs = document.getElementById("no-jobs");
    if (!latestJobs.length) {
      tbody.innerHTML = "";
      noJobs.style.display = "block";
      return;
    }
    noJobs.style.display = "none";

    tbody.innerHTML = latestJobs.map(job => {
      const created = job.created_at.replace("T", " ").slice(0, 19);
      const imgShort = job.docker_image.split("/").pop().slice(0, 28);
      const isTerminal = ["completed", "failed", "cancelled"].includes(job.status);
      const cancelBtn = isTerminal ? "" :
        `<button class="action-btn cancel" onclick="cancelJob('${job.id}', event)">Cancel</button>`;
      const userDot = `<span class="user-dot" style="background:${userColor(job.submitted_by || 'anonymous')}"></span>`;
      return `<tr onclick="showLogs('${job.id}')">
        <td class="mono-sm">${job.id.slice(0, 12)}</td>
        <td>${esc(job.name)}</td>
        <td>${userDot}${esc(job.submitted_by || "anonymous")}</td>
        <td><span class="badge badge-${job.status}">${job.status}</span></td>
        <td>${job.requested_gpus}</td>
        <td class="muted mono-sm">${esc(imgShort)}</td>
        <td class="muted mono-sm">${created}</td>
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

async function cancelJob(jobId, event) {
  event.stopPropagation();
  if (!confirm(`Cancel job ${jobId.slice(0, 12)}?`)) return;
  await apiFetch(`/api/v1/jobs/${jobId}`, { method: "DELETE" });
  refresh();
}

// ─── Logs ─────────────────────────────────────────────────────────────────────

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
    const pre = document.getElementById("log-content");
    if (res.ok) {
      const text = await res.text();
      pre.textContent = text || "(no output yet)";
      pre.scrollTop = pre.scrollHeight;
    } else if (res.status === 404) {
      pre.textContent = "(no logs available yet)";
    }
  } catch (e) {
    console.error("Log load error", e);
  }
}

function closeLog() {
  selectedJobId = null;
  document.getElementById("log-section").style.display = "none";
}

// ─── MLflow ───────────────────────────────────────────────────────────────────

async function loadMLflow() {
  const container = document.getElementById("mlflow-runs");
  if (!container) return;
  try {
    const res = await apiFetch("/api/v1/mlflow/runs");
    if (res.status === 503) {
      container.innerHTML = '<span class="muted">MLflow server starting…</span>';
      return;
    }
    if (!res.ok) return;
    const data = await res.json();
    const runs = data.runs || [];

    // Update MLflow UI link
    const btn = document.getElementById("mlflow-btn");
    if (btn) btn.href = `http://${location.hostname}:5001`;

    if (!runs.length) {
      container.innerHTML = '<span class="muted">No experiment runs yet. Submit a job that logs to MLflow to see results here.</span>';
      return;
    }

    container.innerHTML = `<table class="mlflow-table">
      <thead><tr><th>Run ID</th><th>Experiment</th><th>Status</th><th>Started</th></tr></thead>
      <tbody>${runs.map(r => {
        const info = r.info || {};
        const started = info.start_time ? new Date(info.start_time).toLocaleString() : "—";
        return `<tr>
          <td class="mono-sm">${(info.run_id || "").slice(0, 8)}…</td>
          <td>${esc(info.experiment_id || "—")}</td>
          <td><span class="badge badge-${(info.status || "").toLowerCase()}">${info.status || "—"}</span></td>
          <td class="muted mono-sm">${started}</td>
        </tr>`;
      }).join("")}</tbody>
    </table>`;
  } catch (e) {
    // Silently ignore — MLflow may not be ready
  }
}

// ─── Debug Sessions ───────────────────────────────────────────────────────────

async function loadDebugImages() {
  const sel = document.getElementById("debug-image-select");
  if (!sel) return;
  try {
    const res = await apiFetch("/api/v1/debug/images");
    if (!res.ok) return;
    const images = await res.json();
    if (!images.length) {
      sel.innerHTML = '<option value="">No local Docker images found</option>';
      return;
    }
    sel.innerHTML = images.flatMap(img =>
      img.tags.map(tag => `<option value="${esc(tag)}">${esc(tag)} (${img.size_mb} MB)</option>`)
    ).join("");
  } catch (e) {
    sel.innerHTML = '<option value="">Could not load images</option>';
  }
}

async function launchDebugSession() {
  const sel = document.getElementById("debug-image-select");
  const image = sel ? sel.value : "";
  if (!image) { alert("Select a Docker image first."); return; }

  try {
    const res = await apiFetch("/api/v1/debug/sessions", {
      method: "POST",
      body: JSON.stringify({ image }),
    });
    if (res.status === 409) { alert("Max debug sessions reached. Kill an existing session first."); return; }
    if (!res.ok) { const e = await res.json().catch(() => ({})); alert("Failed to launch debug session: " + (e.detail || res.status)); return; }
    const session = await res.json();
    window.open(session.vscode_url, "_blank");
    await loadDebugSessions();
  } catch (e) {
    console.error("Debug launch error", e);
  }
}

async function loadDebugSessions() {
  const container = document.getElementById("debug-sessions-list");
  if (!container) return;
  try {
    const res = await apiFetch("/api/v1/debug/sessions");
    if (!res.ok) return;
    const sessions = await res.json();
    if (!sessions.length) {
      container.innerHTML = '<p class="muted">No active debug sessions.</p>';
      return;
    }
    container.innerHTML = sessions.map(s =>
      `<div class="debug-card">
        <div class="debug-card-info">
          <span class="debug-image">${esc(s.image.split("/").pop().slice(0, 40))}</span>
          <code class="debug-exec muted">${esc(s.exec_cmd)}</code>
        </div>
        <div class="debug-card-actions">
          <a href="${esc(s.vscode_url)}" target="_blank" class="action-btn">Open VS Code ↗</a>
          <button class="action-btn cancel" onclick="killDebugSession('${s.id}')">Kill</button>
        </div>
      </div>`
    ).join("");
  } catch (e) {
    console.error("Debug sessions error", e);
  }
}

async function killDebugSession(id) {
  await apiFetch(`/api/v1/debug/sessions/${id}`, { method: "DELETE" });
  await loadDebugSessions();
}

// ─── Refresh loop ─────────────────────────────────────────────────────────────

async function refresh() {
  await Promise.all([loadGPUs(), loadJobs()]);
  renderResourcePanel(latestJobs, latestGPUs);
  loadMLflow();
  loadDebugSessions();
  if (selectedJobId) await refreshLogs();
}

// Init
if (getApiKey()) {
  document.getElementById("api-key-status").textContent = "Key set ✓";
} else {
  promptApiKey();
}

// Load debug images once (rarely changes)
loadDebugImages();

refresh();
setInterval(refresh, 3000);
