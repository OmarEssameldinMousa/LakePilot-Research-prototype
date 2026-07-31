"""
LakeGym v5 — Interactive Dashboard (NiceGUI)

Professional research dashboard for:
  • Real-time simulation monitoring
  • Policy selection & comparison
  • Agent decision transparency
  • Experiment management
  • Workload plan loading

Run: python app.py
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any

from nicegui import ui, app

# ── Local imports ──
from simulation import LakeSimulator, WORKLOAD_PROFILES
from policies.base import Action, Observation
from policies import ALL_POLICIES
from experiment import ExperimentManager
from metrics import OfflineMetricsCollector
from agents.model_registry import MODEL_REGISTRY
from training.online_training import (
    OnlineSpecialistTrainer, MultiAgentOnlineTrainer,
    PPOConfig, DQNConfig, RolloutBuffer,
)


# ═══════════════════════════════════════════════════════════════
# GLOBALS
# ═══════════════════════════════════════════════════════════════

sim = LakeSimulator()
collector = OfflineMetricsCollector()
experiment_mgr: Optional[ExperimentManager] = None

# Simulation state
is_running = False
batch_stop_requested = False
current_policy_name = "No_Maintenance"
current_policy = None
sim_speed = 0.3       # seconds between steps
step_count = 0

# Chart data (rolling windows)
MAX_CHART_POINTS = 200
latency_data = deque(maxlen=MAX_CHART_POINTS)
files_data = deque(maxlen=MAX_CHART_POINTS)
reward_data = deque(maxlen=MAX_CHART_POINTS)
cumulative_reward_data = deque(maxlen=MAX_CHART_POINTS)
pruning_data = deque(maxlen=MAX_CHART_POINTS)
util_data = deque(maxlen=MAX_CHART_POINTS)

# Agent decision info
last_agent_info: Dict[str, Any] = {}

# Online training state
online_trainer: Optional[MultiAgentOnlineTrainer] = None
online_training_active = False
online_ep_reward_data = deque(maxlen=MAX_CHART_POINTS)
online_ep_latency_data = deque(maxlen=MAX_CHART_POINTS)
online_ep_pruning_data = deque(maxlen=MAX_CHART_POINTS)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def get_policy_instance(name: str):
    """Create a fresh policy instance from the registry."""
    if name not in ALL_POLICIES:
        return None
    factory = ALL_POLICIES[name]
    return factory() if callable(factory) else factory


def clear_chart_data():
    for d in [latency_data, files_data, reward_data, cumulative_reward_data, pruning_data, util_data]:
        d.clear()


def _make_chart_safe(chart):
    """Patch a Highchart so its .update() swallows RuntimeError from dead clients."""
    _original = chart.update
    def _safe_update():
        try:
            _original()
        except RuntimeError:
            pass
    chart.update = _safe_update
    return chart


# ═══════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════

@ui.page('/')
def main_page():
    global is_running, current_policy_name, current_policy
    global sim_speed, step_count, experiment_mgr

    experiment_mgr = ExperimentManager(sim)

    # Stop interactive simulation if the client disconnects (tab close / refresh)
    # NOTE: Do NOT stop batch experiments on disconnect — websocket hiccups
    # auto-reconnect but the flag stays False, killing data collection.
    async def _on_disconnect():
        global is_running
        if is_running:
            is_running = False
            print('⚠️ Client disconnected — interactive simulation stopped.')
    app.on_disconnect(_on_disconnect)

    # ── Dark theme ──
    ui.dark_mode().enable()
    ui.colors(primary='#3b82f6', secondary='#8b5cf6', accent='#10b981')

    # ════════════════════════════════════════════════════
    # HEADER
    # ════════════════════════════════════════════════════
    with ui.header().classes('bg-gray-900 items-center justify-between px-6'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('storage', size='sm').classes('text-blue-400')
            ui.label('LakeGym v5').classes('text-xl font-bold text-white')
            ui.label('Multi-Agent Data Lake Maintenance').classes('text-sm text-gray-400')
        with ui.row().classes('items-center gap-4'):
            status_badge = ui.badge('IDLE', color='gray').classes('text-xs')
            step_label = ui.label('Step: 0').classes('text-sm text-gray-300')

    # ════════════════════════════════════════════════════
    # LEFT DRAWER — CONTROLS
    # ════════════════════════════════════════════════════
    with ui.left_drawer(value=True).classes('bg-gray-800 p-4').style('width: 280px'):

        ui.label('⚙️ Controls').classes('text-lg font-bold text-white mb-3')

        # Policy selector
        ui.label('Policy').classes('text-xs text-gray-400 mt-2')
        policy_select = ui.select(
            options=sorted(ALL_POLICIES.keys()),
            value=current_policy_name,
        ).classes('w-full')

        # Speed slider
        ui.label('Speed (sec/step)').classes('text-xs text-gray-400 mt-4')
        speed_slider = ui.slider(min=0.05, max=2.0, step=0.05, value=sim_speed).classes('w-full')
        speed_display = ui.label(f'{sim_speed:.2f}s').classes('text-xs text-gray-400')

        def on_speed(e):
            global sim_speed
            sim_speed = e.value
            speed_display.set_text(f'{sim_speed:.2f}s')
        speed_slider.on('update:model-value', on_speed)

        # Ingestion rate
        ui.label('Ingestion (rows/step)').classes('text-xs text-gray-400 mt-4')
        ingestion_slider = ui.slider(min=1, max=30, step=1, value=5).classes('w-full')

        def on_ingestion(e):
            sim.set_ingestion_rate(int(e.value))
        ingestion_slider.on('update:model-value', on_ingestion)

        ui.separator().classes('my-4')

        # ── Online Evaluation Section ──
        ui.label('🧠 Online Evaluation').classes('text-lg font-bold text-white mb-2')

        ol_model_type = ui.select(
            options=sorted(MODEL_REGISTRY.keys()),
            value='attentive_ppo',
            label='Model Type',
        ).classes('w-full')

        ol_mode = ui.select(
            options=['Online Adaptive', 'Offline Frozen'],
            value='Online Adaptive',
            label='Experiment Mode',
        ).classes('w-full')

        ol_workload = ui.select(
            options=['v5_eval (500 steps)', 'v5 (1000 steps)', 'v5_fast (1000 steps)'],
            value='v5_eval (500 steps)',
            label='Eval Workload',
        ).classes('w-full')

        ol_lr = ui.number('Learning Rate', value=3e-4, format='%.1e').classes('w-full')
        ol_episodes = ui.number('Episodes', value=5, min=1, max=50, step=1).classes('w-full')

        # Weights upload — compaction
        ui.label('Compaction Weights (.h5)').classes('text-xs text-gray-400 mt-2')
        _ol_compact_weights = {'path': None}
        ol_compact_upload = ui.upload(
            label='Upload Compaction .h5', auto_upload=True, max_files=1,
        ).classes('w-full').props('accept=.h5,.weights.h5')
        ol_compact_label = ui.label('No weights').classes('text-xs text-gray-500')

        # Weights upload — partition
        ui.label('Partition Weights (.h5)').classes('text-xs text-gray-400 mt-2')
        _ol_partition_weights = {'path': None}
        ol_partition_upload = ui.upload(
            label='Upload Partition .h5', auto_upload=True, max_files=1,
        ).classes('w-full').props('accept=.h5,.weights.h5')
        ol_partition_label = ui.label('No weights').classes('text-xs text-gray-500')

        _weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')

        def _upload_weights(e, path_dict, label_widget, agent_name):
            try:
                content = e.content.read()
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                os.makedirs(_weights_dir, exist_ok=True)
                fname = os.path.join(_weights_dir, f'uploaded_{agent_name}_{ts}.weights.h5')
                with open(fname, 'wb') as f:
                    f.write(content)
                path_dict['path'] = fname
                label_widget.set_text(f'✅ {os.path.basename(fname)}')
            except Exception as ex:
                label_widget.set_text(f'❌ {ex}')

        ol_compact_upload.on('upload', lambda e: _upload_weights(e, _ol_compact_weights, ol_compact_label, 'compact'))
        ol_partition_upload.on('upload', lambda e: _upload_weights(e, _ol_partition_weights, ol_partition_label, 'partition'))

        with ui.row().classes('w-full gap-2 mt-2'):
            ol_start_btn = ui.button('▶ Run', color='teal').classes('flex-1')
            ol_stop_btn = ui.button('■ Stop', color='red').classes('flex-1')
        ol_train_status = ui.label('').classes('text-xs text-gray-500 mt-1')

        ui.separator().classes('my-4')

        # Control buttons
        with ui.row().classes('w-full gap-2'):
            start_btn = ui.button('▶ Start', color='green').classes('flex-1')
            stop_btn = ui.button('■ Stop', color='red').classes('flex-1')

        with ui.row().classes('w-full gap-2 mt-2'):
            reset_btn = ui.button('⟳ Reset', color='orange').classes('flex-1')
            export_btn = ui.button('📥 Export', color='blue').classes('flex-1')

        ui.separator().classes('my-4')

        # Workload plan
        ui.label('📋 Workload Plan').classes('text-xs text-gray-400')
        plan_upload = ui.upload(
            label='Upload JSON',
            auto_upload=True,
            max_files=1,
        ).classes('w-full').props('accept=.json')

        plan_status = ui.label('No plan loaded').classes('text-xs text-gray-500 mt-1')

        ui.separator().classes('my-4')

        # Profile override
        ui.label('Workload Profile Override').classes('text-xs text-gray-400')
        profile_select = ui.select(
            options=['auto'] + list(WORKLOAD_PROFILES.keys()),
            value='auto',
        ).classes('w-full')

        def on_profile(e):
            if e.value == 'auto':
                sim.config.manual_profile = None
            else:
                sim.config.manual_profile = e.value
        profile_select.on('update:model-value', on_profile)

        ui.separator().classes('my-4')

        # Experiment section
        ui.label('🧪 Batch Experiment').classes('text-xs text-gray-400')
        exp_episodes = ui.number('Episodes', value=3, min=1, max=20, step=1).classes('w-full')
        exp_steps = ui.number('Steps/ep', value=1000, min=100, max=5000, step=100).classes('w-full')
        exp_btn = ui.button('Run Batch', color='purple').classes('w-full mt-2')
        exp_progress = ui.label('').classes('text-xs text-gray-500 mt-1')

    # ════════════════════════════════════════════════════
    # MAIN CONTENT
    # ════════════════════════════════════════════════════
    with ui.column().classes('w-full p-6 gap-4'):

        # ── ROW 1: Metric cards ──
        with ui.grid(columns=5).classes('w-full gap-4'):
            with ui.card().classes('flex-1 bg-gray-800 border border-gray-700'):
                ui.label('LATENCY').classes('text-[10px] text-blue-400 font-bold tracking-wide')
                latency_val = ui.label('— ms').classes('text-2xl font-bold text-white')
                ui.label('query response time').classes('text-[10px] text-gray-500')

            with ui.card().classes('flex-1 bg-gray-800 border border-gray-700'):
                ui.label('FILES').classes('text-[10px] text-amber-400 font-bold tracking-wide')
                files_val = ui.label('—').classes('text-2xl font-bold text-white')
                ui.label('data files in table').classes('text-[10px] text-gray-500')

            with ui.card().classes('flex-1 bg-gray-800 border border-gray-700'):
                ui.label('BLOCK UTIL').classes('text-[10px] text-green-400 font-bold tracking-wide')
                util_val = ui.label('— %').classes('text-2xl font-bold text-white')
                ui.label('avg / optimal file size').classes('text-[10px] text-gray-500')

            with ui.card().classes('flex-1 bg-gray-800 border border-gray-700'):
                ui.label('PRUNING').classes('text-[10px] text-purple-400 font-bold tracking-wide')
                pruning_val = ui.label('—').classes('text-2xl font-bold text-white')
                ui.label('partition pruning ratio').classes('text-[10px] text-gray-500')

            with ui.card().classes('flex-1 bg-gray-800 border border-gray-700'):
                ui.label('REWARD').classes('text-[10px] text-emerald-400 font-bold tracking-wide')
                reward_val = ui.label('—').classes('text-2xl font-bold text-white')
                ui.label('cumulative global').classes('text-[10px] text-gray-500')

        # ── ROW 2: Charts ──
        CHART_COMMON = {
            'chart': {'backgroundColor': 'transparent', 'height': 220, 'animation': False},
            'credits': {'enabled': False},
            'legend': {'enabled': False},
            'title': {'style': {'color': '#9ca3af', 'fontSize': '12px'}},
            'xAxis': {'labels': {'style': {'color': '#6b7280'}}, 'gridLineColor': '#374151'},
            'yAxis': {'labels': {'style': {'color': '#6b7280'}}, 'gridLineColor': '#374151'},
        }

        with ui.grid(columns=2).classes('w-full gap-4'):
            with ui.card().classes('bg-gray-800 border border-gray-700 p-2'):
                latency_chart = _make_chart_safe(ui.highchart({
                    **CHART_COMMON,
                    'title': {**CHART_COMMON['title'], 'text': 'Query Latency'},
                    'yAxis': {**CHART_COMMON['yAxis'], 'title': {'text': 'ms', 'style': {'color': '#6b7280'}}},
                    'series': [{'name': 'Latency', 'data': [], 'color': '#60a5fa', 'lineWidth': 1.5}],
                }).classes('w-full'))

            with ui.card().classes('bg-gray-800 border border-gray-700 p-2'):
                files_chart = _make_chart_safe(ui.highchart({
                    **CHART_COMMON,
                    'title': {**CHART_COMMON['title'], 'text': 'File Count'},
                    'yAxis': {**CHART_COMMON['yAxis'], 'title': {'text': 'files', 'style': {'color': '#6b7280'}}},
                    'series': [{'name': 'Files', 'data': [], 'color': '#fbbf24', 'lineWidth': 1.5}],
                }).classes('w-full'))

            with ui.card().classes('bg-gray-800 border border-gray-700 p-2'):
                reward_chart = _make_chart_safe(ui.highchart({
                    **CHART_COMMON,
                    'title': {**CHART_COMMON['title'], 'text': 'Cumulative Reward'},
                    'yAxis': {**CHART_COMMON['yAxis'], 'title': {'text': 'reward', 'style': {'color': '#6b7280'}}},
                    'series': [{'name': 'Reward', 'data': [], 'color': '#34d399', 'lineWidth': 1.5}],
                }).classes('w-full'))

            with ui.card().classes('bg-gray-800 border border-gray-700 p-2'):
                query_chart = _make_chart_safe(ui.highchart({
                    **CHART_COMMON,
                    'chart': {**CHART_COMMON['chart'], 'type': 'pie'},
                    'title': {**CHART_COMMON['title'], 'text': 'Query Distribution'},
                    'plotOptions': {'pie': {
                        'dataLabels': {'enabled': True, 'style': {'color': '#9ca3af', 'fontSize': '10px'}},
                        'colors': ['#60a5fa', '#f472b6', '#a78bfa', '#fbbf24', '#6b7280'],
                    }},
                    'series': [{'name': 'Queries', 'data': [
                        {'name': 'TIME_RANGE', 'y': 20},
                        {'name': 'REGION', 'y': 20},
                        {'name': 'SENSOR', 'y': 20},
                        {'name': 'TYPE', 'y': 20},
                        {'name': 'FULL_SCAN', 'y': 20},
                    ]}],
                }).classes('w-full'))

        # ── ROW 3: Agent Decisions ──
        with ui.card().classes('w-full bg-gray-800 border border-gray-700'):
            ui.label('🤖 Agent Decisions').classes('text-sm font-bold text-white mb-2')
            with ui.row().classes('w-full gap-6'):
                with ui.column().classes('flex-1'):
                    ui.label('Meta-Controller').classes('text-xs text-gray-400 font-bold')
                    meta_label = ui.label('—').classes('text-lg font-bold text-blue-400')
                    meta_reason = ui.label('').classes('text-xs text-gray-500')

                with ui.column().classes('flex-1'):
                    ui.label('Compaction Agent').classes('text-xs text-gray-400 font-bold')
                    compact_label = ui.label('—').classes('text-lg font-bold text-amber-400')
                    compact_conf = ui.label('').classes('text-xs text-gray-500')

                with ui.column().classes('flex-1'):
                    ui.label('Partition Agent').classes('text-xs text-gray-400 font-bold')
                    partition_label = ui.label('—').classes('text-lg font-bold text-purple-400')
                    partition_conf = ui.label('').classes('text-xs text-gray-500')

                with ui.column().classes('flex-1'):
                    ui.label('Current State').classes('text-xs text-gray-400 font-bold')
                    state_label = ui.label('—').classes('text-sm text-white')
                    profile_label = ui.label('').classes('text-xs text-gray-500')

        # ── ROW 4: Evaluation Metrics (expandable) ──
        with ui.expansion('📊 Evaluation Metrics', icon='insights').classes(
            'w-full bg-gray-800 border border-gray-700'
        ):
            with ui.grid(columns=3).classes('w-full gap-4 p-2'):
                with ui.card().classes('bg-gray-800 border border-gray-700 p-2'):
                    train_reward_chart = _make_chart_safe(ui.highchart({
                        **CHART_COMMON,
                        'chart': {**CHART_COMMON['chart'], 'height': 200},
                        'title': {**CHART_COMMON['title'], 'text': 'Episode Reward'},
                        'yAxis': {**CHART_COMMON['yAxis'], 'title': {'text': 'cum. reward', 'style': {'color': '#6b7280'}}},
                        'series': [{'name': 'Reward', 'data': [], 'color': '#22c55e', 'lineWidth': 1.5}],
                    }).classes('w-full'))

                with ui.card().classes('bg-gray-800 border border-gray-700 p-2'):
                    train_latency_chart = _make_chart_safe(ui.highchart({
                        **CHART_COMMON,
                        'chart': {**CHART_COMMON['chart'], 'height': 200},
                        'title': {**CHART_COMMON['title'], 'text': 'Avg Latency'},
                        'yAxis': {**CHART_COMMON['yAxis'], 'title': {'text': 'ms', 'style': {'color': '#6b7280'}}},
                        'series': [{'name': 'Latency', 'data': [], 'color': '#60a5fa', 'lineWidth': 1.5}],
                    }).classes('w-full'))

                with ui.card().classes('bg-gray-800 border border-gray-700 p-2'):
                    train_pruning_chart = _make_chart_safe(ui.highchart({
                        **CHART_COMMON,
                        'chart': {**CHART_COMMON['chart'], 'height': 200},
                        'title': {**CHART_COMMON['title'], 'text': 'Avg Pruning Ratio'},
                        'yAxis': {**CHART_COMMON['yAxis'], 'title': {'text': 'pruning', 'style': {'color': '#6b7280'}}},
                        'series': [{'name': 'Pruning', 'data': [], 'color': '#a78bfa', 'lineWidth': 1.5}],
                    }).classes('w-full'))

        # ── ROW 5: Log ──
        with ui.card().classes('w-full bg-gray-800 border border-gray-700'):
            ui.label('📝 Log').classes('text-sm font-bold text-white mb-1')
            log = ui.log(max_lines=30).classes('w-full h-40 bg-gray-900 text-gray-300 text-xs')

    # ════════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════════
    with ui.footer().classes('bg-gray-900 text-gray-400 text-xs px-6 py-2'):
        with ui.row().classes('w-full justify-between'):
            footer_left = ui.label('Rows: 0 | Size: 0 KB')
            footer_mid = ui.label('Partition: UNPARTITIONED')
            footer_right = ui.label('LakeGym v5 — Research Dashboard')

    # ════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ════════════════════════════════════════════════════

    async def init_sim():
        log.push("🚀 Initializing Spark + Iceberg...")
        msg = await asyncio.to_thread(sim.initialize)
        log.push(msg)
        if "✅" in msg:
            status_badge.set_text("READY")
            status_badge.props('color=green')
        else:
            status_badge.set_text("ERROR")
            status_badge.props('color=red')

    async def run_simulation():
        global is_running, current_policy, step_count

        if not sim.state.is_initialized:
            await init_sim()
            if not sim.state.is_initialized:
                return

        current_policy_name_local = policy_select.value
        current_policy = get_policy_instance(current_policy_name_local)
        if not current_policy:
            log.push(f"❌ Unknown policy: {current_policy_name_local}")
            return

        # Auto-load workload plan if not already active
        if not sim.scenario_manager.is_active:
            default_plan = "workload_plan_v5.json"
            if os.path.exists(default_plan):
                with open(default_plan) as f:
                    plan_data = json.load(f)
                msg = sim.scenario_manager.load_plan(plan_data)
                log.push(f"📋 Auto-loaded: {msg}")

        is_running = True
        status_badge.set_text("RUNNING")
        status_badge.props('color=blue')
        collector.set_policy(current_policy.name)
        log.push(f"▶ Started: {current_policy.name}")

        prev_obs = sim.get_observation()

        while is_running:
            obs = Observation.from_dict(prev_obs)
            action = current_policy.get_action(obs)
            result = await asyncio.to_thread(sim.run_step, action)
            next_obs = sim.get_observation()

            # Record transition
            collector.record_from_result(prev_obs, action, result, next_obs)

            # Update chart data
            step_count += 1
            latency_data.append(result.latency_ms)
            files_data.append(result.file_count)
            cumulative_reward_data.append(result.cumulative_global_reward)
            pruning_data.append(result.partition_pruning_ratio)
            util_data.append(result.block_utilization)

            # Update UI — wrapped to handle client disconnect (tab close/refresh)
            try:
                step_label.set_text(f'Step: {step_count}')
                latency_val.set_text(f'{result.latency_ms:.0f} ms')
                files_val.set_text(str(result.file_count))
                util_val.set_text(f'{result.block_utilization*100:.0f}%')
                pruning_val.set_text(f'{result.partition_pruning_ratio:.2f}')
                reward_val.set_text(f'{result.cumulative_global_reward:+.2f}')

                # Charts
                latency_chart.options['series'][0]['data'] = list(latency_data)
                latency_chart.update()
                files_chart.options['series'][0]['data'] = list(files_data)
                files_chart.update()
                reward_chart.options['series'][0]['data'] = list(cumulative_reward_data)
                reward_chart.update()

                # Query distribution pie
                obs_dict = next_obs
                qd = [
                    {'name': 'TIME_RANGE', 'y': max(obs_dict.get('query_hist_time_range', 0), 0.01)},
                    {'name': 'REGION', 'y': max(obs_dict.get('query_hist_region_filter', 0), 0.01)},
                    {'name': 'SENSOR', 'y': max(obs_dict.get('query_hist_sensor_lookup', 0), 0.01)},
                    {'name': 'TYPE', 'y': max(obs_dict.get('query_hist_type_filter', 0), 0.01)},
                    {'name': 'FULL_SCAN', 'y': max(obs_dict.get('query_hist_full_scan', 0), 0.01)},
                ]
                query_chart.options['series'][0]['data'] = qd
                query_chart.update()

                # Agent decision panel
                info = current_policy.get_info() if hasattr(current_policy, 'get_info') else {}
                delegation = info.get('meta_delegation', None)
                if delegation:
                    meta_label.set_text(f'→ {delegation.upper()}')
                    meta_reason.set_text(info.get('meta_reason', ''))
                    cu = info.get('meta_compact_urgency', 0)
                    pu = info.get('meta_partition_urgency', 0)

                    # Show compaction agent probs
                    cp = info.get('compact_probs')
                    if cp:
                        c_names = ['NOOP', 'C32', 'C64', 'C128']
                        best_ci = max(range(len(cp)), key=lambda i: cp[i])
                        compact_label.set_text(f'{c_names[best_ci]} ({cp[best_ci]*100:.0f}%)')
                        compact_conf.set_text(
                            f'urgency: {cu:.2f} │ ' +
                            ' '.join(f'{c_names[i]}:{cp[i]*100:.0f}%' for i in range(len(cp)))
                        )
                    else:
                        compact_label.set_text(result.action_taken)
                        compact_conf.set_text(f'urgency: {cu:.2f}')

                    # Show partition agent probs
                    pp = info.get('partition_probs')
                    if pp:
                        p_names = ['NOOP', 'HOUR', 'REGION', 'TYPE', 'REMOVE']
                        best_pi = max(range(len(pp)), key=lambda i: pp[i])
                        partition_label.set_text(f'{p_names[best_pi]} ({pp[best_pi]*100:.0f}%)')
                        partition_conf.set_text(
                            f'urgency: {pu:.2f} │ ' +
                            ' '.join(f'{p_names[i]}:{pp[i]*100:.0f}%' for i in range(len(pp)))
                        )
                    else:
                        partition_label.set_text(result.partition_strategy)
                        partition_conf.set_text(f'urgency: {pu:.2f}')
                else:
                    meta_label.set_text(current_policy.name)
                    meta_reason.set_text('')
                    compact_label.set_text(result.action_taken)
                    compact_conf.set_text('')
                    partition_label.set_text(result.partition_strategy)
                    partition_conf.set_text('')

                state_label.set_text(
                    f'Files: {result.file_count} | '
                    f'Util: {result.block_utilization:.2f} | '
                    f'Prune: {result.partition_pruning_ratio:.2f}'
                )
                profile_label.set_text(f'Profile: {result.workload_profile}')

                # Footer
                footer_left.set_text(f'Rows: {result.total_rows:,} | Size: {result.total_size_kb:.0f} KB')
                footer_mid.set_text(f'Partition: {result.partition_strategy} ({result.partition_count} parts)')

                # Log
                if result.action_taken != 'NOOP' or step_count % 10 == 0:
                    log.push(
                        f'[{step_count}] {result.action_taken} | '
                        f'lat={result.latency_ms:.0f}ms files={result.file_count} '
                        f'R={result.global_reward:+.3f} Q={result.query_type}'
                    )
            except RuntimeError:
                # Client disconnected (tab closed/refreshed) — stop loop gracefully
                is_running = False
                print(f"⚠️ Client disconnected at step {step_count}, stopping simulation loop.")
                break

            prev_obs = next_obs
            await asyncio.sleep(sim_speed)

    def stop_simulation():
        global is_running, batch_stop_requested
        is_running = False
        batch_stop_requested = True
        status_badge.set_text("STOPPED")
        status_badge.props('color=orange')
        log.push("■ Stopped")

    async def reset_simulation():
        global step_count
        stop_simulation()
        msg = await asyncio.to_thread(sim.reset)
        log.push(msg)
        step_count = 0
        clear_chart_data()
        collector.clear()
        step_label.set_text('Step: 0')
        latency_val.set_text('— ms')
        files_val.set_text('—')
        util_val.set_text('— %')
        pruning_val.set_text('—')
        reward_val.set_text('—')
        status_badge.set_text("READY")
        status_badge.props('color=green')

    async def export_data():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"results/interactive_{ts}"
        os.makedirs(path, exist_ok=True)
        msg1 = sim.export_history_to_csv(f"{path}/history.csv")
        msg2 = collector.export_transitions(f"{path}/transitions.csv")
        log.push(msg1)
        log.push(msg2)

    def on_plan_upload(e):
        try:
            content = e.content.read()
            plan = json.loads(content)
            msg = sim.scenario_manager.load_plan(plan)
            plan_status.set_text(msg)
            log.push(msg)
        except Exception as ex:
            plan_status.set_text(f'❌ {ex}')

    async def run_batch_experiment():
        global step_count, batch_stop_requested

        if experiment_mgr.is_running:
            log.push("⚠️ Experiment already running")
            return

        stop_simulation()

        policy_name = policy_select.value
        policy = get_policy_instance(policy_name)
        if not policy:
            log.push(f"❌ Unknown policy: {policy_name}")
            return

        if not sim.state.is_initialized:
            await init_sim()

        n_ep = int(exp_episodes.value)
        n_steps = int(exp_steps.value)

        # Load workload plan if not already active
        if not sim.scenario_manager.is_active:
            default_plan = "workload_plan_v5.json"
            if os.path.exists(default_plan):
                with open(default_plan) as f:
                    plan_data = json.load(f)
                msg = sim.scenario_manager.load_plan(plan_data)
                log.push(f"📋 Auto-loaded: {msg}")
            else:
                log.push("⚠️ No workload plan found — using random ingestion/profiles")
        else:
            log.push(f"📋 Using active plan: {sim.scenario_manager.plan_name}")

        log.push(f"🧪 Starting batch: {policy_name} × {n_ep} episodes × {n_steps} steps")
        batch_stop_requested = False
        status_badge.set_text("EXPERIMENT")
        status_badge.props('color=purple')
        exp_progress.set_text("Starting...")

        out_dir = f"results/{policy_name}"

        # Run the experiment step-by-step so the UI stays responsive
        os.makedirs(out_dir, exist_ok=True)

        from metrics import OfflineMetricsCollector as _Collector
        from datetime import datetime as _dt

        timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
        all_transitions = _Collector()
        all_transitions.set_policy(policy.name)
        episode_summaries = []

        try:
            for ep in range(n_ep):
                if batch_stop_requested:
                    log.push("■ Batch stopped by user")
                    break
                await asyncio.to_thread(sim.reset)
                policy.reset()
                ep_collector = _Collector()
                ep_collector.set_policy(policy.name)

                prev_obs = sim.get_observation()
                step_count = 0
                clear_chart_data()

                log.push(f"  Episode {ep+1}/{n_ep} starting...")

                for step in range(1, n_steps + 1):
                    if batch_stop_requested:
                        break
                    obs = Observation.from_dict(prev_obs)
                    action = policy.get_action(obs)
                    result = await asyncio.to_thread(sim.run_step, action)
                    next_obs = sim.get_observation()

                    # Record transition
                    done = (step == n_steps)
                    ep_collector.record_from_result(prev_obs, action, result, next_obs, done)
                    all_transitions.record_from_result(prev_obs, action, result, next_obs, done)

                    # Update chart data
                    step_count += 1
                    latency_data.append(result.latency_ms)
                    files_data.append(result.file_count)
                    cumulative_reward_data.append(result.cumulative_global_reward)
                    pruning_data.append(result.partition_pruning_ratio)

                    # Update UI every 5 steps (keep it snappy)
                    try:
                        if step % 5 == 0 or step == n_steps:
                            pct = ((ep * n_steps + step) / (n_ep * n_steps)) * 100
                            exp_progress.set_text(
                                f"Ep {ep+1}/{n_ep} · Step {step}/{n_steps} · {pct:.0f}%"
                            )
                            step_label.set_text(f'Step: {step_count}')
                            latency_val.set_text(f'{result.latency_ms:.0f} ms')
                            files_val.set_text(str(result.file_count))
                            util_val.set_text(f'{result.block_utilization*100:.0f}%')
                            pruning_val.set_text(f'{result.partition_pruning_ratio:.2f}')
                            reward_val.set_text(f'{result.cumulative_global_reward:+.2f}')

                            latency_chart.options['series'][0]['data'] = list(latency_data)
                            latency_chart.update()
                            files_chart.options['series'][0]['data'] = list(files_data)
                            files_chart.update()
                            reward_chart.options['series'][0]['data'] = list(cumulative_reward_data)
                            reward_chart.update()

                            # Query distribution pie
                            qd = [
                                {'name': 'TIME_RANGE', 'y': max(next_obs.get('query_hist_time_range', 0), 0.01)},
                                {'name': 'REGION', 'y': max(next_obs.get('query_hist_region_filter', 0), 0.01)},
                                {'name': 'SENSOR', 'y': max(next_obs.get('query_hist_sensor_lookup', 0), 0.01)},
                                {'name': 'TYPE', 'y': max(next_obs.get('query_hist_type_filter', 0), 0.01)},
                                {'name': 'FULL_SCAN', 'y': max(next_obs.get('query_hist_full_scan', 0), 0.01)},
                            ]
                            query_chart.options['series'][0]['data'] = qd
                            query_chart.update()

                            footer_left.set_text(f'Rows: {result.total_rows:,} | Size: {result.total_size_kb:.0f} KB')
                            footer_mid.set_text(f'Partition: {result.partition_strategy} ({result.partition_count} parts)')

                            # Agent decision panel
                            b_info = policy.get_info() if hasattr(policy, 'get_info') else {}
                            b_deleg = b_info.get('meta_delegation', None)
                            if b_deleg:
                                meta_label.set_text(f'→ {b_deleg.upper()}')
                                meta_reason.set_text(b_info.get('meta_reason', ''))
                                b_cu = b_info.get('meta_compact_urgency', 0)
                                b_pu = b_info.get('meta_partition_urgency', 0)

                                b_cp = b_info.get('compact_probs')
                                if b_cp:
                                    b_cn = ['NOOP', 'C32', 'C64', 'C128']
                                    b_ci = max(range(len(b_cp)), key=lambda i: b_cp[i])
                                    compact_label.set_text(f'{b_cn[b_ci]} ({b_cp[b_ci]*100:.0f}%)')
                                    compact_conf.set_text(
                                        f'urgency: {b_cu:.2f} │ ' +
                                        ' '.join(f'{b_cn[i]}:{b_cp[i]*100:.0f}%' for i in range(len(b_cp)))
                                    )
                                else:
                                    compact_label.set_text(result.action_taken)
                                    compact_conf.set_text(f'urgency: {b_cu:.2f}')

                                b_pp = b_info.get('partition_probs')
                                if b_pp:
                                    b_pn = ['NOOP', 'HOUR', 'REGION', 'TYPE', 'REMOVE']
                                    b_pi = max(range(len(b_pp)), key=lambda i: b_pp[i])
                                    partition_label.set_text(f'{b_pn[b_pi]} ({b_pp[b_pi]*100:.0f}%)')
                                    partition_conf.set_text(
                                        f'urgency: {b_pu:.2f} │ ' +
                                        ' '.join(f'{b_pn[i]}:{b_pp[i]*100:.0f}%' for i in range(len(b_pp)))
                                    )
                                else:
                                    partition_label.set_text(result.partition_strategy)
                                    partition_conf.set_text(f'urgency: {b_pu:.2f}')
                            else:
                                meta_label.set_text(policy.name)
                                meta_reason.set_text('')
                                compact_label.set_text(result.action_taken)
                                compact_conf.set_text('')
                                partition_label.set_text(result.partition_strategy)
                                partition_conf.set_text('')

                            state_label.set_text(
                                f'Files: {result.file_count} | '
                                f'Util: {result.block_utilization:.2f} | '
                                f'Prune: {result.partition_pruning_ratio:.2f}'
                            )
                            profile_label.set_text(f'Profile: {result.workload_profile}')

                        if step % 20 == 0:
                            log.push(
                                f'[Ep{ep+1} S{step}] {result.action_taken} | '
                                f'lat={result.latency_ms:.0f}ms files={result.file_count} '
                                f'R={result.global_reward:+.3f}'
                            )
                        if step % 100 == 0:
                            ingest_desc = sim.scenario_manager.get_ingestion_description(step) or 'N/A'
                            log.push(
                                f'  📋 S{step}: profile={result.workload_profile} '
                                f'rows={result.rows_ingested} rate={result.ingestion_rate_rows_per_sec:.0f}r/s'
                            )
                            log.push(
                                f'  📦 S{step}: ingestion phase="{ingest_desc}"'
                            )
                    except RuntimeError:
                        pass  # client disconnected — keep collecting data anyway

                    prev_obs = next_obs

                    # Small yield to let UI repaint (to_thread already unblocks the loop,
                    # but this gives NiceGUI a moment to flush updates to the browser)
                    await asyncio.sleep(0.05)

                # Save episode transitions
                ep_path = os.path.join(out_dir, f"transitions_{timestamp}_ep{ep+1}.csv")
                ep_collector.export_transitions(ep_path)

                ep_stats = sim.get_summary_stats()
                ep_stats['episode'] = ep + 1
                episode_summaries.append(ep_stats)

                log.push(
                    f"  ✅ Episode {ep+1} done — "
                    f"avg_lat: {ep_stats.get('avg_latency', 0):.0f}ms, "
                    f"avg_R: {ep_stats.get('avg_global_reward', 0):+.4f}"
                )

            # Save combined transitions
            combined_path = os.path.join(out_dir, f"transitions_{timestamp}.csv")
            all_transitions.export_transitions(combined_path)

            log.push(f"✅ Batch done! {all_transitions.transition_count} transitions → {out_dir}")

        except Exception as e:
            log.push(f"❌ Batch error: {e}")
            import traceback
            traceback.print_exc()

        is_running = False
        exp_progress.set_text('Done')
        status_badge.set_text("READY")
        status_badge.props('color=green')

    # ── Wire events ──
    start_btn.on('click', run_simulation)
    stop_btn.on('click', stop_simulation)
    reset_btn.on('click', reset_simulation)
    export_btn.on('click', export_data)
    plan_upload.on('upload', on_plan_upload)
    exp_btn.on('click', run_batch_experiment)

    # ── Online Evaluation Handler ──
    async def run_online_evaluation():
        global online_trainer, online_training_active

        if online_training_active:
            log.push('⚠️ Evaluation already running')
            return

        if not sim.state.is_initialized:
            await init_sim()
            if not sim.state.is_initialized:
                return

        model_type = ol_model_type.value
        mode = ol_mode.value  # 'Online Adaptive' or 'Offline Frozen'
        is_adaptive = (mode == 'Online Adaptive')
        n_ep = int(ol_episodes.value)
        lr = float(ol_lr.value)
        compact_weights = _ol_compact_weights.get('path')
        partition_weights = _ol_partition_weights.get('path')

        # Parse workload selection
        wl_choice = ol_workload.value
        if 'eval' in wl_choice:
            plan_file = 'workload_plan_v5_eval.json'
        elif 'fast' in wl_choice:
            plan_file = 'workload_plan_v5_fast.json'
        else:
            plan_file = 'workload_plan_v5.json'

        # Load workload plan
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(src_dir, '..'))
        candidates = [
            os.path.join(project_root, plan_file),   # LakeGymLite/
            os.path.join(os.getcwd(), plan_file),     # CWD
            os.path.join(src_dir, plan_file),          # src/
            plan_file,                                  # bare filename
        ]
        plan_path = None
        for c in candidates:
            if os.path.exists(c):
                plan_path = c
                break
        if plan_path is None:
            log.push(f'❌ Workload plan not found: {plan_file}')
            log.push(f'   Searched: {[os.path.normpath(c) for c in candidates]}')
            return

        with open(plan_path) as f:
            plan_data = json.load(f)
        n_steps = plan_data.get('total_steps', 500)

        # Build multi-agent trainer
        try:
            if model_type == 'ddqn':
                dqn_config = DQNConfig(learning_rate=lr)
                online_trainer = MultiAgentOnlineTrainer(
                    model_type=model_type, dqn_config=dqn_config,
                    compact_weights=compact_weights, partition_weights=partition_weights,
                )
            else:
                ppo_config = PPOConfig(actor_lr=lr, rollout_steps=n_steps)
                online_trainer = MultiAgentOnlineTrainer(
                    model_type=model_type, config=ppo_config,
                    compact_weights=compact_weights, partition_weights=partition_weights,
                )
        except Exception as e:
            log.push(f'❌ Trainer init error: {e}')
            import traceback; traceback.print_exc()
            return

        online_training_active = True
        online_ep_reward_data.clear()
        online_ep_latency_data.clear()
        online_ep_pruning_data.clear()

        mode_tag = 'adaptive' if is_adaptive else 'frozen'
        log.push(f'🧠 {mode}: {model_type} × {n_ep} ep × {n_steps} steps [{plan_file}]')
        if compact_weights:
            log.push(f'  📦 Compaction weights: {os.path.basename(compact_weights)}')
        if partition_weights:
            log.push(f'  📦 Partition weights: {os.path.basename(partition_weights)}')
        ol_train_status.set_text(f'Running {mode}...')

        # Results directory
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..',
            'results', f'eval_{mode_tag}_{model_type}'
        )
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Load plan into simulator
        sim.scenario_manager.load_plan(plan_data)
        log.push(f'📋 Loaded: {plan_data.get("name", plan_file)}')

        from metrics import OfflineMetricsCollector
        import numpy as _np

        try:
            for ep in range(n_ep):
                if not online_training_active:
                    log.push('■ Evaluation stopped')
                    break

                # Reset sim and agents for new episode
                await asyncio.to_thread(sim.reset)
                online_trainer.reset_episode()
                sim.scenario_manager.load_plan(plan_data)  # reload plan after reset

                ep_collector = OfflineMetricsCollector()
                ep_collector.set_policy(f'eval_{mode_tag}_{model_type}')
                compact_buffer = RolloutBuffer()
                partition_buffer = RolloutBuffer()

                ep_rewards = []
                ep_latencies = []
                ep_pruning = []
                ep_actions = []
                obs_dict = sim.get_observation()

                log.push(f'  ── Episode {ep+1}/{n_ep} starting ──')

                for step in range(1, n_steps + 1):
                    if not online_training_active:
                        break

                    # Run one multi-agent step.
                    # Frozen mode evaluates deterministically (argmax);
                    # adaptive mode explores (sampling / epsilon-greedy).
                    step_info = await asyncio.to_thread(
                        online_trainer.step, sim, obs_dict, not is_adaptive,
                    )
                    result = step_info['result']
                    next_obs_dict = step_info['next_obs_dict']
                    done = (step == n_steps)

                    # Record transition in specialist's buffer
                    specialist = step_info['specialist']
                    if specialist == 'compaction' and step_info['window'] is not None:
                        compact_buffer.add(
                            step_info['window'], step_info['action_idx'],
                            step_info['reward'], step_info['value'],
                            step_info['log_prob'], done,
                        )
                    elif specialist == 'partition' and step_info['window'] is not None:
                        partition_buffer.add(
                            step_info['window'], step_info['action_idx'],
                            step_info['reward'], step_info['value'],
                            step_info['log_prob'], done,
                        )

                    # Record transition for CSV
                    action_enum = Action[result.action_taken]
                    ep_collector.record_from_result(obs_dict, action_enum, result, next_obs_dict, done)

                    ep_rewards.append(step_info['global_reward'])
                    ep_latencies.append(float(result.latency_ms))
                    ep_pruning.append(float(result.partition_pruning_ratio))
                    ep_actions.append(result.action_taken)
                    obs_dict = next_obs_dict

                    # Live logging
                    try:
                        if step % 10 == 0 or step == n_steps:
                            pct = ((ep * n_steps + step) / (n_ep * n_steps)) * 100
                            ol_train_status.set_text(
                                f'Ep {ep+1}/{n_ep} · S{step}/{n_steps} · {pct:.0f}%'
                            )
                        if step % 50 == 0:
                            log.push(
                                f'  [Ep{ep+1} S{step}] {result.action_taken} '
                                f'({step_info["delegation"]}) | '
                                f'lat={result.latency_ms:.0f}ms files={result.file_count} '
                                f'R={step_info["global_reward"]:+.3f}'
                            )
                    except RuntimeError:
                        pass

                    if step % 5 == 0:
                        await asyncio.sleep(0.02)

                if not online_training_active:
                    break

                # Save episode transitions CSV
                ep_csv = os.path.join(results_dir, f'transitions_{timestamp}_ep{ep+1}.csv')
                ep_collector.export_transitions(ep_csv)

                # Episode stats
                ep_reward_sum = float(_np.sum(ep_rewards))
                ep_avg_lat = float(_np.mean(ep_latencies)) if ep_latencies else 0.0
                ep_avg_prune = float(_np.mean(ep_pruning)) if ep_pruning else 0.0

                # Online Adaptive: update both models
                update_msg = ''
                if is_adaptive:
                    log.push(f'  🔄 Updating model weights...')
                    metrics = await asyncio.to_thread(
                        online_trainer.update_models, compact_buffer, partition_buffer,
                    )
                    c_loss = metrics['compaction'].get('loss', 0.0)
                    p_loss = metrics['partition'].get('loss', 0.0)
                    update_msg = f' c_loss={c_loss:.4f} p_loss={p_loss:.4f}'

                # Update adaptation curve charts
                online_ep_reward_data.append(ep_reward_sum)
                online_ep_latency_data.append(ep_avg_lat)
                online_ep_pruning_data.append(ep_avg_prune)

                try:
                    train_reward_chart.options['series'][0]['data'] = list(online_ep_reward_data)
                    train_reward_chart.update()
                    train_latency_chart.options['series'][0]['data'] = list(online_ep_latency_data)
                    train_latency_chart.update()
                    train_pruning_chart.options['series'][0]['data'] = list(online_ep_pruning_data)
                    train_pruning_chart.update()
                except RuntimeError:
                    pass

                from collections import Counter
                log.push(
                    f'  ✅ Ep{ep+1} — R={ep_reward_sum:.2f} '
                    f'avg_lat={ep_avg_lat:.0f}ms avg_prune={ep_avg_prune:.3f}'
                    f'{update_msg}'
                )
                log.push(f'     actions: {dict(Counter(ep_actions))}')
                log.push(f'     CSV → {os.path.basename(ep_csv)}')

                online_trainer.record_episode({
                    'episode': ep + 1, 'mode': mode_tag,
                    'reward_sum': ep_reward_sum, 'avg_latency': ep_avg_lat,
                    'avg_pruning': ep_avg_prune, 'actions': dict(Counter(ep_actions)),
                })
                await asyncio.sleep(0.1)

            # Save weights if adaptive
            if is_adaptive and online_trainer is not None:
                save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
                os.makedirs(save_dir, exist_ok=True)
                c_path = os.path.join(save_dir, f'compact_online_{model_type}.weights.h5')
                p_path = os.path.join(save_dir, f'partition_online_{model_type}.weights.h5')
                online_trainer.save_weights(c_path, p_path)
                log.push(f'✅ Weights saved: {os.path.basename(c_path)}, {os.path.basename(p_path)}')

            log.push(f'✅ Evaluation done! CSVs → {results_dir}')

        except Exception as e:
            log.push(f'❌ Evaluation error: {e}')
            import traceback; traceback.print_exc()

        online_training_active = False
        ol_train_status.set_text('Done')

    def stop_online_evaluation():
        global online_training_active
        online_training_active = False
        ol_train_status.set_text('Stopped')
        log.push('■ Evaluation stopped')

    ol_start_btn.on('click', run_online_evaluation)
    ol_stop_btn.on('click', stop_online_evaluation)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    # Suppress stale-client RuntimeErrors from NiceGUI's internal prop watchers
    import logging
    logging.getLogger('nicegui').setLevel(logging.ERROR)

    ui.run(
        title="LakeGym v5",
        host="0.0.0.0",
        port=8080,
        dark=True,
        reload=False,
        storage_secret='lakegym-v5',
    )


if __name__ in {'__main__', '__mp_main__'}:
    main()
