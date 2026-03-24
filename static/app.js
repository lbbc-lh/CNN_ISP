const fileInput = document.getElementById("file-input");
const analyzeButton = document.getElementById("analyze-button");
const statusText = document.getElementById("status-text");
const sourceImage = document.getElementById("source-image");
const overlayImage = document.getElementById("overlay-image");
const structureOverlayImage = document.getElementById("structure-overlay-image");
const sourceMeta = document.getElementById("source-meta");
const requestId = document.getElementById("request-id");
const finalScore = document.getElementById("final-score");
const missingRegions = document.getElementById("missing-regions");
const suggestionsList = document.getElementById("suggestions-list");
const weightsGrid = document.getElementById("weights-grid");
const globalMetrics = document.getElementById("global-metrics");
const detectedRegions = document.getElementById("detected-regions");
const regionMetricsTable = document.querySelector("#region-metrics-table tbody");
const semanticStructureTable = document.querySelector("#semantic-structure-table tbody");
const rawJson = document.getElementById("raw-json");
const tabs = document.querySelectorAll(".tab");
const tabPanels = document.querySelectorAll(".tab-panel");

const maskEls = {
  person: document.getElementById("mask-person"),
  sky: document.getElementById("mask-sky"),
  vegetation: document.getElementById("mask-vegetation"),
  background: document.getElementById("mask-background"),
};

const structureMaskEls = {
  flat: document.getElementById("mask-flat"),
  edge: document.getElementById("mask-edge"),
  texture: document.getElementById("mask-texture"),
};

let selectedFile = null;

function setActiveTab(tabName) {
  tabs.forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.tab === tabName);
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.panel === tabName);
  });
}

function toDataUrl(base64Data) {
  return `data:image/png;base64,${base64Data}`;
}

function formatNumber(value) {
  if (typeof value !== "number") return String(value);
  return value.toFixed(2);
}

function renderMetricGrid(target, values) {
  target.innerHTML = "";
  for (const [key, value] of Object.entries(values)) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = formatNumber(value);
    target.append(dt, dd);
  }
}

function renderPills(target, values, className = "") {
  target.innerHTML = "";
  if (!values.length) {
    const empty = document.createElement("span");
    empty.className = "pill success";
    empty.textContent = "None";
    target.appendChild(empty);
    return;
  }
  for (const value of values) {
    const span = document.createElement("span");
    span.className = `pill ${className}`.trim();
    span.textContent = value;
    target.appendChild(span);
  }
}

function renderSuggestions(values) {
  suggestionsList.innerHTML = "";
  if (!values.length) {
    const item = document.createElement("li");
    item.textContent = "No suggestions.";
    suggestionsList.appendChild(item);
    return;
  }
  for (const value of values) {
    const item = document.createElement("li");
    item.textContent = value;
    suggestionsList.appendChild(item);
  }
}

function renderRegionMetrics(regionMetrics) {
  regionMetricsTable.innerHTML = "";
  for (const [region, metrics] of Object.entries(regionMetrics)) {
    const row = document.createElement("tr");
    const values = [
      region,
      metrics.valid ? "yes" : "no",
      formatNumber(metrics.coverage_ratio),
      formatNumber(metrics.sharpness),
      formatNumber(metrics.noise),
      formatNumber(metrics.exposure),
      formatNumber(metrics.score),
    ];
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = value;
      row.appendChild(td);
    }
    regionMetricsTable.appendChild(row);
  }
}

function renderSemanticStructureTable(target, metricsByRegion) {
  target.innerHTML = "";
  for (const [region, metrics] of Object.entries(metricsByRegion)) {
    const row = document.createElement("tr");
    const values = [
      region,
      metrics.valid ? "yes" : "no",
      formatNumber(metrics.coverage_ratio),
      formatNumber(metrics.flat_noise),
      formatNumber(metrics.edge_sharpness),
      formatNumber(metrics.texture_clarity),
    ];
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = value;
      row.appendChild(td);
    }
    target.appendChild(row);
  }
}

function renderDetectedRegions(items) {
  detectedRegions.innerHTML = "";
  if (!items.length) {
    detectedRegions.textContent = "No valid semantic region detected.";
    return;
  }

  for (const item of items) {
    const card = document.createElement("article");
    card.className = "region-item";
    card.innerHTML = `
      <strong>${item.label}</strong>
      <div>coverage: ${formatNumber(item.coverage_ratio)}</div>
      <div>weight: ${formatNumber(item.weight)}</div>
      <div>score: ${formatNumber(item.metrics.score)}</div>
    `;
    detectedRegions.appendChild(card);
  }
}

function renderResults(result) {
  requestId.textContent = result.request_id || "N/A";
  finalScore.textContent = formatNumber(result.final_score);
  renderPills(missingRegions, result.missing_regions || [], "error");
  renderSuggestions(result.suggestions || []);
  renderMetricGrid(weightsGrid, result.effective_weights || {});
  renderMetricGrid(globalMetrics, result.global_metrics || {});
  renderRegionMetrics(result.region_metrics || {});
  renderSemanticStructureTable(semanticStructureTable, result.semantic_structure_metrics || {});
  renderDetectedRegions(result.detected_regions || []);
  rawJson.textContent = JSON.stringify(result, null, 2);

  overlayImage.src = toDataUrl(result.visualizations.overlay_base64);
  structureOverlayImage.src = toDataUrl(result.structure_visualizations.overlay_base64);
  for (const [region, el] of Object.entries(maskEls)) {
    const value = result.visualizations.mask_base64[region];
    el.src = toDataUrl(value);
  }
  for (const [region, el] of Object.entries(structureMaskEls)) {
    const value = result.structure_visualizations.mask_base64[region];
    el.src = toDataUrl(value);
  }
}

async function analyze() {
  if (!selectedFile) return;

  analyzeButton.disabled = true;
  statusText.textContent = "Analyzing image...";

  try {
    const base64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = String(reader.result);
        resolve(result.split(",", 2)[1]);
      };
      reader.onerror = () => reject(new Error("Unable to read file"));
      reader.readAsDataURL(selectedFile);
    });

    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Analysis failed");
    }

    renderResults(payload);
    statusText.textContent = "Analysis completed.";
  } catch (error) {
    statusText.textContent = `Error: ${error.message}`;
  } finally {
    analyzeButton.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  analyzeButton.disabled = !selectedFile;
  if (!selectedFile) {
    statusText.textContent = "Choose an image to begin.";
    return;
  }

  sourceMeta.textContent = `${selectedFile.name} · ${(selectedFile.size / 1024).toFixed(1)} KB`;
  statusText.textContent = "Ready to analyze.";

  const reader = new FileReader();
  reader.onload = () => {
    sourceImage.src = reader.result;
  };
  reader.readAsDataURL(selectedFile);
});

analyzeButton.addEventListener("click", analyze);

tabs.forEach((tab) => {
  tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
});
