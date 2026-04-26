const state = {
  latestTraining: null,
  latestSimulation: null,
  latestComparison: null,
  latestExplanation: null,
  algorithms: [],
  currentEnvironment: {
    price_level: "medium",
    temperature_level: "normal",
    presence: "home",
  },
  charts: {},
};

const algorithmDefaults = {
  random: 30,
  rule_based: 30,
  q_learning: 140,
  sarsa: 150,
  dqn: 170,
  ppo: 160,
};

const chartPalette = {
  teal: "#0f8b6d",
  orange: "#df6d3b",
  gold: "#f1c453",
  ink: "#19352a",
  sky: "#1f6e8c",
  soft: "rgba(15, 139, 109, 0.16)",
};

const els = {
  algorithmSelect: document.getElementById("algorithmSelect"),
  trainingEpisodesInput: document.getElementById("trainingEpisodesInput"),
  simulationEpisodesInput: document.getElementById("simulationEpisodesInput"),
  seedInput: document.getElementById("seedInput"),
  priceLevelSelect: document.getElementById("priceLevelSelect"),
  temperatureLevelSelect: document.getElementById("temperatureLevelSelect"),
  presenceSelect: document.getElementById("presenceSelect"),
  applyEnvironmentBtn: document.getElementById("applyEnvironmentBtn"),
  trainBtn: document.getElementById("trainBtn"),
  simulateBtn: document.getElementById("simulateBtn"),
  compareBtn: document.getElementById("compareBtn"),
  explainBtn: document.getElementById("explainBtn"),
  refreshResultsBtn: document.getElementById("refreshResultsBtn"),
  statusPill: document.getElementById("statusPill"),
  statusText: document.getElementById("statusText"),
  metricReward: document.getElementById("metricReward"),
  metricRewardSub: document.getElementById("metricRewardSub"),
  metricCost: document.getElementById("metricCost"),
  metricCostSub: document.getElementById("metricCostSub"),
  metricComfort: document.getElementById("metricComfort"),
  metricComfortSub: document.getElementById("metricComfortSub"),
  metricBattery: document.getElementById("metricBattery"),
  metricBatterySub: document.getElementById("metricBatterySub"),
  trainingSummary: document.getElementById("trainingSummary"),
  simulationSummary: document.getElementById("simulationSummary"),
  environmentSummary: document.getElementById("environmentSummary"),
  behaviorSummary: document.getElementById("behaviorSummary"),
  explanationPanel: document.getElementById("explanationPanel"),
  currentPriceLevel: document.getElementById("currentPriceLevel"),
  currentTemperatureLevel: document.getElementById("currentTemperatureLevel"),
  currentPresence: document.getElementById("currentPresence"),
  environmentImpactList: document.getElementById("environmentImpactList"),
  environmentFeedback: document.getElementById("environmentFeedback"),
  insightsGrid: document.getElementById("insightsGrid"),
  comparisonTable: document.querySelector("#comparisonTable tbody"),
  actionLogTable: document.querySelector("#actionLogTable tbody"),
};

document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  initCharts();
  await fetchResults();
});

function setupEventListeners() {
  els.algorithmSelect.addEventListener("change", handleAlgorithmChange);
  els.trainBtn.addEventListener("click", handleTrain);
  els.simulateBtn.addEventListener("click", handleSimulate);
  els.compareBtn.addEventListener("click", handleCompare);
  els.explainBtn.addEventListener("click", handleExplain);
  els.refreshResultsBtn.addEventListener("click", fetchResults);
  els.applyEnvironmentBtn.addEventListener("click", handleApplyEnvironment);
}

function handleAlgorithmChange() {
  const algorithm = els.algorithmSelect.value;
  const defaultEpisodes = algorithmDefaults[algorithm] || 100;
  els.trainingEpisodesInput.value = defaultEpisodes;
}

els.priceLevelSelect.addEventListener("change", renderEnvironmentImpactFromSelection);
els.temperatureLevelSelect.addEventListener("change", renderEnvironmentImpactFromSelection);
els.presenceSelect.addEventListener("change", renderEnvironmentImpactFromSelection);

function parseSeed() {
  const rawValue = els.seedInput.value.trim();
  if (!rawValue) {
    return null;
  }
  const parsed = Number(rawValue);
  return Number.isNaN(parsed) ? null : parsed;
}

function numericValue(input, fallback) {
  const parsed = Number(input.value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function setBusy(isBusy, label = "Running experiment...") {
  [els.trainBtn, els.simulateBtn, els.compareBtn, els.explainBtn, els.refreshResultsBtn, els.applyEnvironmentBtn].forEach((button) => {
    button.disabled = isBusy;
  });
  if (isBusy) {
    setStatus("loading", label);
  }
}

function setStatus(kind, message) {
  els.statusPill.className = `status-pill ${kind}`;
  els.statusPill.textContent =
    kind === "loading" ? "Working..." : kind === "success" ? "Success" : kind === "error" ? "Attention" : "Ready";
  els.statusText.textContent = message;
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });

  if (!response.ok) {
    let detail = "Request failed.";
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch (error) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  return response.json();
}

async function handleApplyEnvironment() {
  setBusy(true, "Applying new environment settings...");
  try {
    const payload = {
      price_level: els.priceLevelSelect.value,
      temperature_level: els.temperatureLevelSelect.value,
      presence: els.presenceSelect.value,
    };
    state.currentEnvironment = await apiRequest("/set-environment", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderEnvironmentState();
    setStatus("success", "Environment settings were applied successfully.");
    els.environmentFeedback.textContent =
      `${capitalizeWord(state.currentEnvironment.price_level)} price, ${capitalizeWord(state.currentEnvironment.temperature_level)} weather, and ${capitalizeWord(state.currentEnvironment.presence)} presence are now active for the next simulation.`;
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(false);
  }
}

async function handleTrain() {
  const algorithm = els.algorithmSelect.value;
  const episodes = numericValue(els.trainingEpisodesInput, algorithmDefaults[algorithm] || 100);
  setBusy(true, `Training ${prettyAlgorithm(algorithm)} for ${episodes} episodes...`);

  try {
    const payload = {
      algorithm,
      episodes,
      seed: parseSeed(),
    };
    state.latestTraining = await apiRequest("/train", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderTraining();
    syncRewardChart();
    setStatus("success", `${state.latestTraining.algorithm_display_name} training finished successfully.`);
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(false);
  }
}

async function handleSimulate() {
  const algorithm = els.algorithmSelect.value;
  const episodes = numericValue(els.simulationEpisodesInput, 3);
  setBusy(true, `Simulating ${prettyAlgorithm(algorithm)} for ${episodes} episodes...`);

  try {
    const payload = {
      algorithm,
      episodes,
      seed: parseSeed(),
    };
    state.latestSimulation = await apiRequest("/simulate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderSimulation();
    renderEnvironmentState();
    syncMetricCards();
    syncRewardChart();
    setStatus("success", `${state.latestSimulation.algorithm_display_name} simulation completed.`);
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(false);
  }
}

async function handleCompare() {
  const trainingEpisodes = numericValue(els.trainingEpisodesInput, 140);
  const simulationEpisodes = numericValue(els.simulationEpisodesInput, 3);
  setBusy(true, `Comparing all algorithms with ${trainingEpisodes} training episodes...`);

  try {
    const payload = {
      episodes_per_algorithm: simulationEpisodes,
      training_episodes: trainingEpisodes,
      seed: parseSeed(),
    };
    state.latestComparison = await apiRequest("/compare", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderComparison();
    renderEnvironmentState();
    setStatus("success", `${state.latestComparison.best_algorithm_display_name} is leading the comparison.`);
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(false);
  }
}

async function handleExplain() {
  const source = state.latestComparison ? "comparison" : "simulation";
  setBusy(true, `Generating AI explanation from the latest ${source}...`);

  try {
    state.latestExplanation = await apiRequest("/explain", {
      method: "POST",
      body: JSON.stringify({ source }),
    });
    renderExplanation();
    setStatus("success", "AI explanation generated.");
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(false);
  }
}

async function fetchResults() {
  setBusy(true, "Loading latest project results...");
  try {
    const results = await apiRequest("/results");
    state.algorithms = results.available_algorithms || [];
    state.latestTraining = results.latest_training;
    state.latestSimulation = results.latest_simulation;
    state.latestComparison = results.latest_comparison;
    state.latestExplanation = results.latest_explanation;
    state.currentEnvironment = results.current_environment || state.currentEnvironment;

    populateAlgorithms();
    renderTraining();
    renderSimulation();
    renderComparison();
    renderExplanation();
    renderEnvironmentState();
    syncMetricCards();
    syncRewardChart();
    setStatus("success", "Dashboard synced with FastAPI successfully.");
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    setBusy(false);
  }
}

function populateAlgorithms() {
  if (!state.algorithms.length) {
    return;
  }

  const currentValue = els.algorithmSelect.value;
  els.algorithmSelect.innerHTML = "";

  state.algorithms.forEach((algorithm) => {
    const option = document.createElement("option");
    option.value = algorithm;
    option.textContent = prettyAlgorithm(algorithm);
    els.algorithmSelect.appendChild(option);
  });

  els.algorithmSelect.value = currentValue && state.algorithms.includes(currentValue) ? currentValue : state.algorithms[0];
  handleAlgorithmChange();
}

function prettyAlgorithm(value) {
  return value
    .split("_")
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

function renderTraining() {
  if (!state.latestTraining) {
    els.trainingSummary.textContent = "No training has been run yet.";
    return;
  }

  const training = state.latestTraining;
  els.trainingSummary.textContent =
    `${training.algorithm_display_name} trained for ${training.episodes} episode(s). ` +
    `Average reward: ${formatNumber(training.average_reward)}. ` +
    `Best episode reward: ${formatNumber(training.best_episode_reward)}. ` +
    training.message;
}

function renderSimulation() {
  if (!state.latestSimulation) {
    els.simulationSummary.textContent = "No simulation has been run yet.";
    els.behaviorSummary.textContent = "Run a simulation to see how the agent adapted.";
    renderSimulationInsights([]);
    renderEmptyActionLog();
    updateLineChart(state.charts.temperatureChart, [], "No simulation data");
    updateLineChart(state.charts.costChart, [], "No simulation data");
    return;
  }

  const simulation = state.latestSimulation;
  els.simulationSummary.textContent = simulation.summary;
  els.behaviorSummary.textContent = simulation.behavior_highlights?.join(" ") || "No behavior highlights yet.";
  renderSimulationInsights(simulation.simulation_insights || []);
  renderActionLog(simulation.action_log);
  updateLineChart(state.charts.temperatureChart, simulation.temperature_history, "Temperature");
  updateLineChart(state.charts.costChart, simulation.cost_history, "Energy Cost");
}

function renderComparison() {
  if (!state.latestComparison) {
    els.comparisonTable.innerHTML = `<tr><td colspan="7" class="empty-cell">No comparison results yet.</td></tr>`;
    updateBarChart(state.charts.comparisonChart, [], []);
    return;
  }

  const rows = state.latestComparison.compared_algorithms
    .map(
      (item, index) => `
        <tr>
          <td><span class="rank-pill">#${index + 1}</span></td>
          <td>${item.display_name}</td>
          <td>${formatNumber(item.average_reward)}</td>
          <td>${formatNumber(item.average_cost)}</td>
          <td>${formatNumber(item.average_comfort_score)}</td>
          <td>${formatNumber(item.average_battery_usage)}</td>
          <td>${item.notes}</td>
        </tr>
      `
    )
    .join("");
  els.comparisonTable.innerHTML = rows;

  updateBarChart(
    state.charts.comparisonChart,
    state.latestComparison.compared_algorithms.map((item) => item.display_name),
    state.latestComparison.compared_algorithms.map((item) => item.average_reward)
  );
}

function renderExplanation() {
  if (!state.latestExplanation) {
    els.explanationPanel.innerHTML =
      '<p class="muted">Run a comparison or simulation, then click “Generate AI Explanation”.</p>';
    return;
  }

  const explanation = state.latestExplanation;
  els.explanationPanel.innerHTML = `
    <div class="explanation-line">
      <strong>Best Algorithm</strong>
      <p>${explanation.best_algorithm}</p>
    </div>
    <div class="explanation-line">
      <strong>Why It Performed Best</strong>
      <p>${explanation.why_it_performed_best}</p>
    </div>
    <div class="explanation-line">
      <strong>How Environment Affected Behavior</strong>
      <p>${explanation.environment_effect}</p>
    </div>
  `;
}

function renderEnvironmentState() {
  const environment = state.latestSimulation?.environment_settings || state.currentEnvironment;
  if (!environment) {
    return;
  }

  els.priceLevelSelect.value = environment.price_level;
  els.temperatureLevelSelect.value = environment.temperature_level;
  els.presenceSelect.value = environment.presence;

  els.currentPriceLevel.textContent = capitalizeWord(environment.price_level);
  els.currentTemperatureLevel.textContent = capitalizeWord(environment.temperature_level);
  els.currentPresence.textContent = capitalizeWord(environment.presence);

  els.environmentSummary.textContent =
    `Current environment: electricity price is ${environment.price_level}, outside temperature is ${environment.temperature_level}, and the user is ${environment.presence}.`;
  renderEnvironmentImpact(environment);
}

function renderEnvironmentImpactFromSelection() {
  renderEnvironmentImpact({
    price_level: els.priceLevelSelect.value,
    temperature_level: els.temperatureLevelSelect.value,
    presence: els.presenceSelect.value,
  });
}

function renderEnvironmentImpact(environment) {
  const impactItems = buildEnvironmentImpactItems(environment);
  els.environmentImpactList.innerHTML = impactItems
    .map(
      (item) => `
        <article class="impact-item">
          <span class="impact-icon">${item.icon}</span>
          <p>${item.text}</p>
        </article>
      `
    )
    .join("");
}

function renderSimulationInsights(insights) {
  if (!insights.length) {
    els.insightsGrid.innerHTML = `
      <article class="insight-card neutral">
        <span class="insight-icon">ℹ️</span>
        <div>
          <strong>Waiting for simulation</strong>
          <p>Run a simulation to see how the agent reacted to the selected environment.</p>
        </div>
      </article>
    `;
    return;
  }

  els.insightsGrid.innerHTML = insights
    .map((insight) => {
      const meta = getInsightMeta(insight);
      return `
        <article class="insight-card ${meta.tone}">
          <span class="insight-icon">${meta.icon}</span>
          <div>
            <strong>${meta.title}</strong>
            <p>${insight}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function getInsightMeta(insight) {
  const normalized = insight.toLowerCase();
  if (normalized.includes("cost") || normalized.includes("expensive") || normalized.includes("high")) {
    if (normalized.includes("decreased") || normalized.includes("reduce")) {
      return { icon: "⚡", tone: "positive", title: "Cost Strategy" };
    }
    return { icon: "💰", tone: "warning", title: "Cost Pressure" };
  }
  if (normalized.includes("battery")) {
    return { icon: "🔋", tone: "positive", title: "Battery Behavior" };
  }
  if (normalized.includes("heater") || normalized.includes("heating") || normalized.includes("cold")) {
    return { icon: "🔥", tone: "warning", title: "Heating Reaction" };
  }
  if (normalized.includes("cooling") || normalized.includes("hot")) {
    return { icon: "❄️", tone: "positive", title: "Cooling Reaction" };
  }
  if (normalized.includes("home") || normalized.includes("away") || normalized.includes("comfort")) {
    return { icon: "🏠", tone: "positive", title: "Comfort Choice" };
  }
  return { icon: "ℹ️", tone: "neutral", title: "Simulation Insight" };
}

function buildEnvironmentImpactItems(environment) {
  const items = [];

  if (environment.price_level === "high") {
    items.push({
      icon: "⚡",
      text: "High electricity price -> agent will try to reduce energy usage and rely more on battery.",
    });
  } else if (environment.price_level === "low") {
    items.push({
      icon: "⚡",
      text: "Low electricity price -> agent can use devices more freely and charge battery with less penalty.",
    });
  } else {
    items.push({
      icon: "⚡",
      text: "Medium electricity price -> agent balances normal energy use with cost control.",
    });
  }

  if (environment.temperature_level === "cold") {
    items.push({
      icon: "❄️",
      text: "Cold weather -> agent will prioritize heating to keep comfort.",
    });
  } else if (environment.temperature_level === "hot") {
    items.push({
      icon: "🔥",
      text: "Hot weather -> agent will use cooling more often.",
    });
  } else {
    items.push({
      icon: "🌤️",
      text: "Normal weather -> agent can focus on balanced heating and cooling decisions.",
    });
  }

  if (environment.presence === "away") {
    items.push({
      icon: "🏠",
      text: "User is away -> agent focuses more on saving cost than comfort.",
    });
  } else {
    items.push({
      icon: "🏠",
      text: "User is home -> agent prioritizes comfort.",
    });
  }

  return items;
}

function syncMetricCards() {
  const simulation = state.latestSimulation;
  if (!simulation) {
    els.metricReward.textContent = "--";
    els.metricCost.textContent = "--";
    els.metricComfort.textContent = "--";
    els.metricBattery.textContent = "--";
    return;
  }

  els.metricReward.textContent = formatNumber(simulation.total_reward);
  els.metricRewardSub.textContent = `${simulation.algorithm_display_name} latest episode`;
  els.metricCost.textContent = `$${formatNumber(simulation.energy_cost)}`;
  els.metricCostSub.textContent = `Average: $${formatNumber(simulation.average_cost)}`;
  els.metricComfort.textContent = formatNumber(simulation.comfort_score);
  els.metricComfortSub.textContent = `Average: ${formatNumber(simulation.average_comfort_score)}`;
  els.metricBattery.textContent = formatNumber(simulation.battery_usage);
  els.metricBatterySub.textContent = `Average: ${formatNumber(simulation.average_battery_usage)}`;
}

function syncRewardChart() {
  if (state.latestTraining?.reward_over_episodes?.length) {
    updateLineChart(
      state.charts.rewardChart,
      state.latestTraining.reward_over_episodes,
      `${state.latestTraining.algorithm_display_name} Training Reward`
    );
    return;
  }

  if (state.latestSimulation?.reward_over_episodes?.length) {
    updateLineChart(
      state.charts.rewardChart,
      state.latestSimulation.reward_over_episodes,
      `${state.latestSimulation.algorithm_display_name} Evaluation Reward`
    );
    return;
  }

  updateLineChart(state.charts.rewardChart, [], "No reward history");
}

function renderActionLog(logRows) {
  if (!logRows?.length) {
    renderEmptyActionLog();
    return;
  }

  els.actionLogTable.innerHTML = logRows
    .map(
      (row) => `
        <tr>
          <td>${row.time}</td>
          <td>
            <div class="state-chip-group">
              <span class="tag">${row.state.room_temperature_category}</span>
              <span class="tag alt">${row.state.electricity_price_category}</span>
              <span class="tag">${row.state.human_presence}</span>
              <span class="tag alt">${row.state.battery_level_category}</span>
              <span class="tag">${row.state.time_of_day}</span>
            </div>
          </td>
          <td>
            ${humanizeAction(row.action)}
            <span class="small-note">${renderDeviceFlags(row.devices)}</span>
          </td>
          <td>${formatNumber(row.reward)}</td>
          <td>$${formatNumber(row.cost)}</td>
          <td>${formatNumber(row.current_temperature)}°C</td>
          <td>$${formatNumber(row.electricity_price)}</td>
        </tr>
      `
    )
    .join("");
}

function renderEmptyActionLog() {
  els.actionLogTable.innerHTML =
    '<tr><td colspan="7" class="empty-cell">Run a simulation to see the action log.</td></tr>';
}

function renderDeviceFlags(devices) {
  const active = Object.entries(devices)
    .filter(([, enabled]) => enabled)
    .map(([key]) => key.replaceAll("_", " "));
  return active.length ? `Devices: ${active.join(", ")}` : "Devices: idle";
}

function humanizeAction(action) {
  return action
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function capitalizeWord(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function initCharts() {
  state.charts.temperatureChart = createLineChart("temperatureChart", chartPalette.teal, "Temperature");
  state.charts.costChart = createLineChart("costChart", chartPalette.orange, "Energy Cost");
  state.charts.rewardChart = createLineChart("rewardChart", chartPalette.sky, "Reward");
  state.charts.comparisonChart = createBarChart("comparisonChart");
}

function baseChartOptions() {
  return {
    maintainAspectRatio: false,
    responsive: true,
    plugins: {
      legend: {
        labels: {
          color: "#183328",
          font: {
            family: "Manrope",
            weight: "700",
          },
        },
      },
      tooltip: {
        backgroundColor: "rgba(24, 51, 40, 0.94)",
        titleFont: { family: "Manrope", weight: "800" },
        bodyFont: { family: "Manrope" },
        padding: 12,
      },
    },
    scales: {
      x: {
        grid: { color: "rgba(24, 51, 40, 0.08)" },
        ticks: { color: "#5f746b", font: { family: "Manrope" } },
      },
      y: {
        grid: { color: "rgba(24, 51, 40, 0.08)" },
        ticks: { color: "#5f746b", font: { family: "Manrope" } },
      },
    },
  };
}

function createLineChart(canvasId, color, label) {
  const context = document.getElementById(canvasId);
  return new Chart(context, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label,
          data: [],
          borderColor: color,
          backgroundColor: chartPalette.soft,
          borderWidth: 3,
          fill: true,
          tension: 0.34,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
      ],
    },
    options: baseChartOptions(),
  });
}

function createBarChart(canvasId) {
  const context = document.getElementById(canvasId);
  return new Chart(context, {
    type: "bar",
    data: {
      labels: [],
      datasets: [
        {
          label: "Average Reward",
          data: [],
          backgroundColor: [
            "#0f8b6d",
            "#df6d3b",
            "#1f6e8c",
            "#f1c453",
            "#1c4b43",
            "#6f857b",
          ],
          borderRadius: 14,
        },
      ],
    },
    options: {
      ...baseChartOptions(),
      plugins: {
        ...baseChartOptions().plugins,
        legend: { display: false },
      },
    },
  });
}

function updateLineChart(chart, values, label) {
  const points = values || [];
  chart.data.labels = points.map((_, index) => index + 1);
  chart.data.datasets[0].label = label;
  chart.data.datasets[0].data = points;
  chart.update();
}

function updateBarChart(chart, labels, values) {
  chart.data.labels = labels;
  chart.data.datasets[0].data = values;
  chart.update();
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(2);
}
