/* ============================================================================
   LakePilot — interactivity: nav, tabs, dynamic tables, heatmap, lightbox
   ========================================================================== */

/* ── sticky nav shadow + mobile toggle ── */
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => nav.classList.toggle('scrolled', window.scrollY > 12));
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');
navToggle.addEventListener('click', () => navLinks.classList.toggle('open'));
navLinks.querySelectorAll('a').forEach(a => a.addEventListener('click', () => navLinks.classList.remove('open')));

/* ── tabs ── */
document.querySelectorAll('[data-tabs]').forEach(group => {
  const scope = group.parentElement;
  group.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      group.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      scope.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      scope.querySelector(`[data-panel="${btn.dataset.tab}"]`).classList.add('active');
    });
  });
});

/* ── reveal on scroll ── */
const revIO = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); revIO.unobserve(e.target); } });
}, { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach(el => revIO.observe(el));

/* ── build ranking table ── */
(function () {
  const tb = document.querySelector('#tbl-ranking tbody');
  DB.ranking.forEach(r => {
    const tr = document.createElement('tr');
    if (r.kind === 'rl') tr.className = 'rl-row';
    if (r.kind === 'bad') tr.className = 'bad-row';
    tr.innerHTML = `
      <td>${r.rank}</td>
      <td class="cell-code">${r.policy} ${r.kind === 'rl' ? '<span class="tag rl">RL</span>' : ''}</td>
      <td class="num">${r.reward.toFixed(4)}</td>
      <td class="num">${r.latency.toLocaleString()}</td>
      <td class="num">${r.files}</td>
      <td class="num">${r.pruning.toFixed(3)}</td>`;
    tb.appendChild(tr);
  });
})();

/* ── action-space table ── */
(function () {
  const tb = document.querySelector('#tbl-actions tbody');
  DB.actionSpace.forEach(a => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class="cell-code">${a.action}</td><td>${a.type}</td><td class="num">${a.cost.toFixed(1)}</td>`;
    tb.appendChild(tr);
  });
})();

/* ── query-profile table (header + rows) ── */
(function () {
  const head = document.querySelector('#tbl-profiles thead tr');
  DB.profiles.cols.forEach(c => { const th = document.createElement('th'); th.className = 'num'; th.textContent = c; head.appendChild(th); });
  const tb = document.querySelector('#tbl-profiles tbody');
  DB.profiles.rows.forEach(r => {
    const tr = document.createElement('tr');
    const max = Math.max(...r.v);
    tr.innerHTML = `<td class="cell-code">${r.name}</td>` +
      r.v.map(v => `<td class="num" ${v === max ? 'style="font-weight:700;color:var(--red-dark)"' : ''}>${v.toFixed(2)}</td>`).join('');
    tb.appendChild(tr);
  });
})();

/* ── consistency table ── */
(function () {
  const tb = document.querySelector('#tbl-consistency tbody');
  DB.consistency.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><b>${r.metric}</b></td><td class="cell-code">${r.a}</td><td class="cell-code">${r.m}</td><td class="cell-code">${r.d}</td>`;
    tb.appendChild(tr);
  });
})();

/* ── training feature tables + config ── */
(function () {
  const fill = (sel, rows) => {
    const tb = document.querySelector(sel); if (!tb) return;
    rows.forEach((r, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td class="num" style="color:var(--ink-3)">${i + 1}</td><td class="cell-code">${r.f}</td><td>${r.norm}</td>`;
      tb.appendChild(tr);
    });
  };
  fill('#tbl-feat-c tbody', DB.features.compact);
  fill('#tbl-feat-p tbody', DB.features.partition);

  const cfg = (sel, rows) => {
    const dl = document.querySelector(sel); if (!dl) return;
    rows.forEach(([k, v]) => { dl.innerHTML += `<dt>${k}</dt><dd>${v}</dd>`; });
  };
  cfg('#cfg-ppo', DB.trainCfg.ppo);
  cfg('#cfg-dqn', DB.trainCfg.dqn);

  // per-architecture offline config + blurbs
  const oa = DB.offlineByArch;
  cfg('#cfg-off-attentive', oa.attentive.cfg);
  cfg('#cfg-off-mlp',       oa.mlp.cfg);
  cfg('#cfg-off-ddqn',      oa.ddqn.cfg);
  const setText = (id, t) => { const el = document.getElementById(id); if (el) el.textContent = t; };
  setText('blurb-oa', oa.attentive.blurb);
  setText('blurb-om', oa.mlp.blurb);
  setText('blurb-od', oa.ddqn.blurb);
  if (oa.ddqn.note) setText('ddqn-note', '⚠ ' + oa.ddqn.note);

  const orw = document.querySelector('#tbl-offline-reward tbody');
  if (orw) DB.offlineReward.forEach(([s, w, cap]) => {
    orw.innerHTML += `<tr><td><b>${s}</b></td><td class="num cell-code">${w}</td><td>${cap}</td></tr>`;
  });
})();

/* ── pruning heatmap ── */
(function () {
  const p = DB.pruning;
  const host = document.getElementById('heatmap');
  if (!host) return;
  let html = '<table class="heatmap"><thead><tr><th></th>' +
    p.cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  p.grid.forEach((row, ri) => {
    html += `<tr><th style="text-align:right">${p.rows[ri]}</th>`;
    row.forEach(v => {
      const bg = v === 0 ? '#F4F2EC' : mix(v);
      const fg = v > 0.45 ? '#fff' : (v === 0 ? '#B7B3A8' : '#1B3139');
      html += `<td><div class="cell" style="background:${bg};color:${fg}">${v === 0 ? '—' : Math.round(v*100)+'%'}</div></td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  host.innerHTML = html;
  function mix(v) {
    // light → Databricks navy ramp
    const t = v;
    const r = Math.round(214 + t * (24 - 214));
    const g = Math.round(225 + t * (49 - 225));
    const b = Math.round(232 + t * (57 - 232));
    return `rgb(${r},${g},${b})`;
  }
})();

/* ── figure gallery ── */
(function () {
  const figs = [
    ['workload_phases', 'Workload phase schedule', 'Query-profile phases across both v5 plans.'],
    ['reward_decomposition', 'Reward decomposition', 'Compact / partition / global reward per key policy.'],
    ['heuristic_baselines', 'Heuristic baseline comparison', 'All 14 baselines ranked by reward, latency, files.'],
    ['compaction_target_size', 'Target-size comparison', 'Rolling 20-step mean for C32 / C64 / C128.'],
    ['rl_three_way', 'Three-way RL comparison', 'Frozen agents on four metrics.'],
    ['action_distribution', 'Action distribution', 'Per-agent action share over 5 episodes.'],
    ['frozen_consistency', 'Frozen consistency', '5-episode spread on the 500-step eval workload.'],
    ['frozen_per_episode', 'Per-episode metrics', 'Mean metric per frozen evaluation episode.'],
    ['adaptive_online_learning', 'Adaptive online learning', 'Reward trajectories under online updates.'],
    ['adaptive_per_episode', 'Adaptive per-episode', 'Mean metric per adaptive episode.'],
    ['adaptive_action_shift', 'Adaptive action shift', 'How action mix shifts during online learning.'],
    ['frozen_vs_adaptive', 'Frozen vs adaptive', 'Per-episode reward comparison by mode.'],
    ['radar_profile', 'Multi-metric radar', 'Normalised four-metric performance profile.'],
    ['cross_arch_variability', 'Cross-architecture variability', 'Box + strip plots over 5 episodes.'],
    ['pruning_matrix', 'Pruning matrix', 'Query type × partition strategy heatmap.'],
    ['cost_analysis', 'Cost breakdown', 'Estimated cloud cost per evaluation episode.'],
  ];
  const host = document.getElementById('gallery');
  figs.forEach(([f, title, sub]) => {
    const src = `assets/figures/${f}.png`;
    const div = document.createElement('div');
    div.className = 'figure';
    div.dataset.full = src;
    div.innerHTML = `<img loading="lazy" src="${src}" alt="${title}"><div class="cap"><b>${title}</b><span>${sub}</span></div>`;
    host.appendChild(div);
  });
})();

/* ── file-pruning tile visualiser ── */
(function () {
  const grid = document.getElementById('tile-grid');
  const toolbar = document.getElementById('prune-toolbar');
  const readout = document.getElementById('prune-readout');
  if (!grid) return;

  const N = 100;
  const P = DB.pruning;                 // rows = query types, cols = strategies
  let queryIdx = 0;                     // TIME_RANGE
  let stratIdx = 3;                     // NONE (unpartitioned) to start

  // build 100 tiles
  for (let i = 0; i < N; i++) { const t = document.createElement('div'); t.className = 'tile'; grid.appendChild(t); }
  const tiles = [...grid.children];

  // toolbar: query-type group + strategy group
  toolbar.innerHTML =
    `<span class="cell-code" style="font-size:.78rem;color:var(--ink-3)">Query:</span>` +
    P.rows.map((q, i) => `<button class="tab-btn" data-q="${i}" style="padding:6px 12px;font-size:.8rem">${q}</button>`).join('') +
    `<span class="cell-code" style="font-size:.78rem;color:var(--ink-3);margin-left:8px">Partition:</span>` +
    P.cols.map((s, i) => `<button class="tab-btn" data-s="${i}" style="padding:6px 12px;font-size:.8rem">${s}</button>`).join('');

  function render() {
    const prune = P.grid[queryIdx][stratIdx];        // fraction skipped
    const scanned = Math.round(N * (1 - prune));
    // deterministic spread of scanned tiles
    tiles.forEach((t, i) => {
      const isScan = (i % N) < scanned;
      t.className = 'tile ' + (isScan ? 'scan' : 'pruned');
    });
    readout.innerHTML =
      `<div><b style="color:var(--red)">${scanned}</b><span style="color:var(--ink-3)">files scanned</span></div>` +
      `<div><b style="color:var(--mlp)">${N - scanned}</b><span style="color:var(--ink-3)">files pruned</span></div>` +
      `<div><b>${Math.round(prune * 100)}%</b><span style="color:var(--ink-3)">pruning ratio</span></div>` +
      (prune === 0 && stratIdx !== 3
        ? `<div style="align-self:center;color:var(--red-dark);font-weight:600;font-size:.85rem">⚠ misaligned — this layout prunes nothing for this query</div>`
        : (queryIdx === 4
            ? `<div style="align-self:center;color:var(--ink-3);font-size:.85rem">FULL_SCAN is irreducible under every strategy</div>`
            : ''));
    // active states
    toolbar.querySelectorAll('[data-q]').forEach(b => b.classList.toggle('active', +b.dataset.q === queryIdx));
    toolbar.querySelectorAll('[data-s]').forEach(b => b.classList.toggle('active', +b.dataset.s === stratIdx));
  }
  toolbar.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.q !== undefined) queryIdx = +b.dataset.q;
    if (b.dataset.s !== undefined) stratIdx = +b.dataset.s;
    render();
  });
  render();
})();

/* ── lightbox (event-delegated, covers gallery + inline figures) ── */
(function () {
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lightbox-img');
  document.addEventListener('click', e => {
    const fig = e.target.closest('[data-full]');
    if (fig) { img.src = fig.dataset.full; lb.classList.add('open'); }
  });
  lb.addEventListener('click', () => lb.classList.remove('open'));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') lb.classList.remove('open'); });
})();
