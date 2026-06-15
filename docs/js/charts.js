/* ============================================================================
   LakePilot — Chart.js renderings of the paper's quantitative results
   ========================================================================== */

const FONT = "Inter, sans-serif";
Chart.defaults.font.family = FONT;
Chart.defaults.color = '#4A5C63';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.legend.labels.padding = 16;

const C = DB.color;
const grid = { color: '#EEEBE4', drawTicks: false };
const tooltip = {
  backgroundColor: '#1B3139', padding: 12, cornerRadius: 8,
  titleFont: { weight: '700' }, bodyFont: { weight: '500' }, displayColors: true,
  boxPadding: 5,
};

/* ── lazy chart builder: only draw when scrolled into view ── */
const builders = {};
function whenVisible(id, build) {
  builders[id] = build;
  const el = document.getElementById(id);
  if (!el) return;
  const io = new IntersectionObserver((entries, obs) => {
    entries.forEach(e => {
      if (e.isIntersecting) { build(); obs.unobserve(e.target); }
    });
  }, { threshold: 0.15 });
  io.observe(el);
}

/* ── burst-phase shaded band plugin (uses DB.problem.burst) ── */
const burstBand = {
  id: 'burstBand',
  beforeDraw(chart) {
    const b = DB.problem.burst;
    const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
    if (!x) return;
    const x0 = x.getPixelForValue(b[0]);
    const x1 = x.getPixelForValue(b[1]);
    ctx.save();
    ctx.fillStyle = 'rgba(239,159,39,.12)';
    ctx.fillRect(x0, top, x1 - x0, bottom - top);
    ctx.fillStyle = '#B8841A';
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillText('ingestion burst', x0 + 6, top + 13);
    ctx.restore();
  },
};

/* ── 0 · The problem — real degradation (file count + latency) ── */
function problemChart(canvasId, nmData, rlData, yTitle, fillNm) {
  const p = DB.problem;
  new Chart(document.getElementById(canvasId), {
    type: 'line',
    data: {
      labels: p.steps,
      datasets: [
        { label: 'No maintenance', data: nmData, borderColor: C.red,
          backgroundColor: fillNm ? 'rgba(255,54,33,.10)' : C.red,
          borderWidth: 2.5, fill: fillNm, tension: .25, pointRadius: 0, pointHoverRadius: 5 },
        { label: 'AttentivePPO', data: rlData, borderColor: C.attentive,
          backgroundColor: 'rgba(24,95,165,.08)',
          borderWidth: 2.5, fill: true, tension: .25, pointRadius: 0, pointHoverRadius: 5 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: { ...tooltip, callbacks: { title: items => `Step ${items[0].label}` } },
      },
      scales: {
        x: { grid: { display: false }, title: { display: true, text: 'Simulation step' },
             ticks: { maxTicksLimit: 6, font: { size: 10 } } },
        y: { grid, title: { display: true, text: yTitle }, beginAtZero: true },
      },
    },
    plugins: [burstBand],
  });
}
whenVisible('chart-problem-files', () =>
  problemChart('chart-problem-files', DB.problem.nmFiles, DB.problem.rlFiles, 'Parquet files', true));
whenVisible('chart-problem-latency', () =>
  problemChart('chart-problem-latency', DB.problem.nmLat, DB.problem.rlLat, 'Latency (ms)', true));

/* ── 0b · Planner cost scatter (latency vs file count, real data) ── */
whenVisible('chart-planner-scatter', () => {
  const p = DB.problem;
  const pts = (files, lat) => files.map((f, i) => ({ x: f, y: lat[i] }));
  new Chart(document.getElementById('chart-planner-scatter'), {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'No maintenance', data: pts(p.nmFiles, p.nmLat),
          backgroundColor: 'rgba(255,54,33,.55)', borderColor: C.red, pointRadius: 4 },
        { label: 'AttentivePPO', data: pts(p.rlFiles, p.rlLat),
          backgroundColor: 'rgba(24,95,165,.6)', borderColor: C.attentive, pointRadius: 4 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: { ...tooltip, callbacks: { label: c => `${c.parsed.x} files → ${Math.round(c.parsed.y)} ms` } },
      },
      scales: {
        x: { grid, title: { display: true, text: 'Parquet file count' }, beginAtZero: true },
        y: { grid, title: { display: true, text: 'Query latency (ms)' }, beginAtZero: true },
      },
    },
  });
});

/* ── 1 · Reward decomposition (stacked-ish grouped bars) ── */
whenVisible('chart-decomp', () => {
  const d = DB.decomposition;
  new Chart(document.getElementById('chart-decomp'), {
    type: 'bar',
    data: {
      labels: d.map(x => x.policy),
      datasets: [
        { label: 'Compact reward',  data: d.map(x => x.compact),  backgroundColor: C.attentive, borderRadius: 3 },
        { label: 'Partition reward',data: d.map(x => x.partition),backgroundColor: C.amber,     borderRadius: 3 },
        { label: 'Global reward',   data: d.map(x => x.global),   backgroundColor: C.red,       borderRadius: 3 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      plugins: { tooltip, legend: { position: 'top' } },
      scales: {
        x: { grid, title: { display: true, text: 'Mean reward' }, ticks: { font: { size: 11 } } },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } },
      },
    },
  });
});

/* ── 2 · Normalised radar ── */
whenVisible('chart-radar', () => {
  const n = DB.normalised;
  new Chart(document.getElementById('chart-radar'), {
    type: 'radar',
    data: {
      labels: n.metrics,
      datasets: [
        ds('AttentivePPO', n.attentive, C.attentive),
        ds('MLP-PPO',      n.mlp,       C.mlp),
        ds('DDQN',         n.ddqn,      C.ddqn),
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { tooltip, legend: { position: 'top' } },
      scales: { r: {
        suggestedMin: 0, suggestedMax: 1,
        grid: { color: '#E3E1DA' }, angleLines: { color: '#E3E1DA' },
        pointLabels: { font: { size: 11, weight: '600' }, color: '#1B3139' },
        ticks: { backdropColor: 'transparent', stepSize: 0.25, font: { size: 9 } },
      } },
    },
  });
  function ds(label, data, color) {
    return { label, data, borderColor: color, backgroundColor: color + '26',
             pointBackgroundColor: color, borderWidth: 2, pointRadius: 3 };
  }
});

/* ── 3 · Three-way reward with error bars (custom) ── */
whenVisible('chart-threeway', () => {
  const t = DB.threeway;
  const errBar = {
    id: 'errBar',
    afterDatasetsDraw(chart) {
      const { ctx, scales: { x, y } } = chart;
      const meta = chart.getDatasetMeta(0);
      ctx.save(); ctx.strokeStyle = '#1B3139'; ctx.lineWidth = 1.5;
      meta.data.forEach((bar, i) => {
        const sd = t.rewardSd[i];
        const top = y.getPixelForValue(t.reward[i] + sd);
        const bot = y.getPixelForValue(t.reward[i] - sd);
        ctx.beginPath();
        ctx.moveTo(bar.x, top); ctx.lineTo(bar.x, bot);
        ctx.moveTo(bar.x - 6, top); ctx.lineTo(bar.x + 6, top);
        ctx.moveTo(bar.x - 6, bot); ctx.lineTo(bar.x + 6, bot);
        ctx.stroke();
      });
      ctx.restore();
    },
  };
  new Chart(document.getElementById('chart-threeway'), {
    type: 'bar',
    data: { labels: t.labels, datasets: [{
      label: 'Mean global reward', data: t.reward,
      backgroundColor: [C.attentive, C.mlp, C.ddqn], borderRadius: 5, maxBarThickness: 90,
    }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { ...tooltip, callbacks: { label: c => `Reward ${c.parsed.y.toFixed(4)} ± ${t.rewardSd[c.dataIndex]}` } },
      },
      scales: { y: { grid, beginAtZero: true, suggestedMax: 0.3 }, x: { grid: { display: false }, ticks: { font: { weight: '600' } } } },
    },
    plugins: [errBar],
  });
});

/* ── 4 · Action distribution (stacked horizontal) ── */
whenVisible('chart-actions', () => {
  const a = DB.actions;
  const cols = ['#A8A69C','#185FA5','#2E77BD','#7DA9D6','#1D9E75','#3BBF95','#62D0AE','#E8593C'];
  new Chart(document.getElementById('chart-actions'), {
    type: 'bar',
    data: {
      labels: ['AttentivePPO', 'MLP-PPO', 'DDQN'],
      datasets: a.labels.map((lab, i) => ({
        label: lab,
        data: [a.attentive[i], a.mlp[i], a.ddqn[i]],
        backgroundColor: cols[i], borderRadius: 2,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      plugins: {
        tooltip: { ...tooltip, callbacks: { label: c => `${c.dataset.label}: ${c.parsed.x}%` } },
        legend: { position: 'bottom', labels: { font: { size: 10 }, padding: 9 } },
      },
      scales: {
        x: { stacked: true, grid, max: 100, title: { display: true, text: '% of steps' } },
        y: { stacked: true, grid: { display: false }, ticks: { font: { weight: '600' } } },
      },
    },
  });
});

/* ── 5 · Compaction target-size (dual axis) ── */
whenVisible('chart-target', () => {
  const t = DB.targetSize;
  new Chart(document.getElementById('chart-target'), {
    type: 'bar',
    data: {
      labels: t.labels,
      datasets: [
        { label: 'Reward', data: t.reward, backgroundColor: C.red, borderRadius: 4, yAxisID: 'y', order: 2 },
        { label: 'Latency (ms)', data: t.latency, type: 'line', borderColor: C.navy, backgroundColor: C.navy, yAxisID: 'y1', tension: .3, pointRadius: 5, order: 1 },
        { label: 'File count', data: t.files, type: 'line', borderColor: C.amber, backgroundColor: C.amber, yAxisID: 'y1', borderDash: [5,4], tension: .3, pointRadius: 5, order: 1 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { tooltip, legend: { position: 'top' } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { weight: '600' } } },
        y:  { position: 'left', grid, title: { display: true, text: 'Reward' }, beginAtZero: true },
        y1: { position: 'right', grid: { display: false }, title: { display: true, text: 'Latency / files' }, beginAtZero: true },
      },
    },
  });
});

/* ── 6 · Adaptive learning curve with ceilings ── */
whenVisible('chart-adaptive', () => {
  const a = DB.adaptive;
  const ceiling = (val, color) => ({
    label: '', data: a.episodes.map(() => val), borderColor: color, borderDash: [4,4],
    borderWidth: 1, pointRadius: 0, fill: false, tension: 0,
  });
  new Chart(document.getElementById('chart-adaptive'), {
    type: 'line',
    data: {
      labels: a.episodes,
      datasets: [
        line('AttentivePPO', a.attentive, C.attentive),
        line('MLP-PPO',      a.mlp,       C.mlp),
        line('DDQN',         a.ddqn,      C.ddqn),
        ceiling(a.ceiling.attentive, C.attentive),
        ceiling(a.ceiling.mlp,       C.mlp),
        ceiling(a.ceiling.ddqn,      C.ddqn),
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        tooltip, legend: { position: 'top', labels: { filter: i => i.text !== '' } },
      },
      scales: {
        x: { grid: { display: false } },
        y: { grid, title: { display: true, text: 'Mean global reward' }, suggestedMin: 0.16, suggestedMax: 0.24 },
      },
    },
  });
  function line(label, data, color) {
    return { label, data, borderColor: color, backgroundColor: color,
             borderWidth: 2.5, tension: .3, pointRadius: 4, pointHoverRadius: 6 };
  }
});

/* ── 7 · Cloud cost (stacked) ── */
whenVisible('chart-cost', () => {
  const c = DB.cost;
  new Chart(document.getElementById('chart-cost'), {
    type: 'bar',
    data: {
      labels: c.labels,
      datasets: [
        { label: 'Query',       data: c.query,   backgroundColor: C.red,   borderRadius: 2 },
        { label: 'Compaction',  data: c.compact, backgroundColor: C.mlp,   borderRadius: 2 },
        { label: 'Storage',     data: c.storage, backgroundColor: C.navy,  borderRadius: 2 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: { ...tooltip, callbacks: {
          label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`,
          footer: items => `Total $${c.total[items[0].dataIndex].toFixed(2)} · ${c.saving[items[0].dataIndex]}`,
        } },
      },
      scales: {
        x: { stacked: true, grid: { display: false }, ticks: { font: { size: 10 }, maxRotation: 30, minRotation: 30 } },
        y: { stacked: true, grid, title: { display: true, text: 'Cost ($ / episode)' } },
      },
    },
  });
});
