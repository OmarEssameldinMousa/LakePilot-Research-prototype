"""
LakeGym v5 — Simulation Engine

Core simulation backend that manages a real Apache Iceberg table via Spark.
Each simulation step: ingest → execute action → run query → compute reward → observe.

Key v5 changes over LakeGymLite v4:
  • 8-column table schema (events) with region, event_type, humidity
  • 5 query templates (TIME_RANGE, REGION_FILTER, SENSOR_LOOKUP, TYPE_FILTER, FULL_SCAN)
  • Workload profiles that shift over time
  • 4 partition strategies (hour, region, event_type, remove)
  • PAYLOAD_SIZE=100 so each row ≈ 200 bytes, ROWS_PER_FILE=50 → ~8 KB/file
  • Three reward signals: compaction, partition, global
  • Query history tracking for partition agent observation
  • Partition pruning ratio computed deterministically
"""

import os
import time
import random
import string
import json
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import IntEnum
from threading import Lock
from collections import deque

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import stddev, col

from policies.base import Action, Observation, ACTION_COSTS, MAINTENANCE_ACTIONS


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

OPTIMAL_FILE_SIZE_KB = 64       # Research-scale optimal file size


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════

class PartitionStrategy(IntEnum):
    """Current partition specification on the Iceberg table."""
    UNPARTITIONED = 0
    HOUR          = 1
    REGION        = 2
    EVENT_TYPE    = 3


class QueryType(IntEnum):
    """Query workload types."""
    TIME_RANGE     = 0
    REGION_FILTER  = 1
    SENSOR_LOOKUP  = 2
    TYPE_FILTER    = 3
    FULL_SCAN      = 4


QUERY_TYPE_NAMES = {
    QueryType.TIME_RANGE:    "TIME_RANGE",
    QueryType.REGION_FILTER: "REGION_FILTER",
    QueryType.SENSOR_LOOKUP: "SENSOR_LOOKUP",
    QueryType.TYPE_FILTER:   "TYPE_FILTER",
    QueryType.FULL_SCAN:     "FULL_SCAN",
}


# ═══════════════════════════════════════════════════════════════
# WORKLOAD PROFILES
# ═══════════════════════════════════════════════════════════════

WORKLOAD_PROFILES: Dict[str, Dict[QueryType, float]] = {
    "time_heavy": {
        QueryType.TIME_RANGE:    0.60,
        QueryType.REGION_FILTER: 0.10,
        QueryType.SENSOR_LOOKUP: 0.10,
        QueryType.TYPE_FILTER:   0.10,
        QueryType.FULL_SCAN:     0.10,
    },
    "region_heavy": {
        QueryType.TIME_RANGE:    0.10,
        QueryType.REGION_FILTER: 0.60,
        QueryType.SENSOR_LOOKUP: 0.10,
        QueryType.TYPE_FILTER:   0.10,
        QueryType.FULL_SCAN:     0.10,
    },
    "type_heavy": {
        QueryType.TIME_RANGE:    0.10,
        QueryType.REGION_FILTER: 0.10,
        QueryType.SENSOR_LOOKUP: 0.10,
        QueryType.TYPE_FILTER:   0.60,
        QueryType.FULL_SCAN:     0.10,
    },
    "mixed": {
        QueryType.TIME_RANGE:    0.20,
        QueryType.REGION_FILTER: 0.20,
        QueryType.SENSOR_LOOKUP: 0.20,
        QueryType.TYPE_FILTER:   0.20,
        QueryType.FULL_SCAN:     0.20,
    },
    "scan_heavy": {
        QueryType.TIME_RANGE:    0.10,
        QueryType.REGION_FILTER: 0.10,
        QueryType.SENSOR_LOOKUP: 0.10,
        QueryType.TYPE_FILTER:   0.10,
        QueryType.FULL_SCAN:     0.60,
    },
}

# Pruning matrix: (QueryType, PartitionStrategy) → base pruning ratio
# Only non-zero entries are listed.
_BASE_PRUNING: Dict[Tuple[QueryType, PartitionStrategy], float] = {
    (QueryType.TIME_RANGE,    PartitionStrategy.HOUR):       0.92,   # ~23/24 hours pruned
    (QueryType.SENSOR_LOOKUP, PartitionStrategy.HOUR):       0.75,   # time filter helps
    (QueryType.REGION_FILTER, PartitionStrategy.REGION):     0.67,   # 1/3 regions
    (QueryType.TYPE_FILTER,   PartitionStrategy.EVENT_TYPE): 0.67,   # 1/3 types
}

# Multiplier applied to every non-zero pruning value (revision Phase 3
# sensitivity analysis). 1.0 reproduces the published matrix exactly.
PRUNING_SCALE = float(os.environ.get('LAKEGYM_PRUNING_SCALE', '1.0'))


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class SimulatorState:
    """Internal mutable state of the simulation."""
    step_count: int = 0
    total_rows: int = 0
    current_latency: float = 0.0
    file_count: int = 0
    total_size_kb: float = 0.0
    partition_strategy: PartitionStrategy = PartitionStrategy.UNPARTITIONED
    last_action: Action = Action.NOOP
    is_initialized: bool = False

    # Maintenance tracking
    last_compact_step: int = 0
    last_partition_step: int = 0

    # Partition coverage tracking
    rows_at_partition_change: int = 0
    compacted_after_partition: bool = True  # starts true (no partition to invalidate)

    # Current workload
    current_profile: str = "mixed"
    last_query_type: Optional[QueryType] = None


@dataclass
class StepResult:
    """Result from a single simulation step — used for CSV export."""
    step: int
    timestamp: str
    rows_ingested: int
    total_rows: int
    latency_ms: float
    file_count: int
    total_size_kb: float
    partition_strategy: str
    partition_count: int
    action_taken: str
    action_success: bool
    ingestion_rate_rows_per_sec: float

    # Advanced metrics
    block_utilization: float = 0.0
    file_size_skew_kb: float = 0.0
    partition_skew: float = 0.0
    partition_pruning_ratio: float = 0.0

    # Three reward signals
    compact_reward: float = 0.0
    partition_reward: float = 0.0
    global_reward: float = 0.0
    cumulative_global_reward: float = 0.0

    # Query info
    query_type: str = ""
    workload_profile: str = ""

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'timestamp': self.timestamp,
            'rows_ingested': self.rows_ingested,
            'total_rows': self.total_rows,
            'latency_ms': round(self.latency_ms, 4),
            'file_count': self.file_count,
            'total_size_kb': round(self.total_size_kb, 4),
            'partition_strategy': self.partition_strategy,
            'partition_count': self.partition_count,
            'action_taken': self.action_taken,
            'action_success': self.action_success,
            'ingestion_rate_rows_per_sec': round(self.ingestion_rate_rows_per_sec, 4),
            'block_utilization': round(self.block_utilization, 6),
            'file_size_skew_kb': round(self.file_size_skew_kb, 6),
            'partition_skew': round(self.partition_skew, 6),
            'partition_pruning_ratio': round(self.partition_pruning_ratio, 6),
            'compact_reward': round(self.compact_reward, 6),
            'partition_reward': round(self.partition_reward, 6),
            'global_reward': round(self.global_reward, 6),
            'cumulative_global_reward': round(self.cumulative_global_reward, 6),
            'query_type': self.query_type,
            'workload_profile': self.workload_profile,
            'error': self.error,
        }


@dataclass
class SimulationConfig:
    """Dynamic configuration tweakable from UI."""
    ingestion_rate_rows: int = 5
    manual_override_active: bool = False
    manual_query_type: Optional[str] = None
    manual_profile: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# SCENARIO MANAGER (workload plan)
# ═══════════════════════════════════════════════════════════════

class ScenarioManager:
    """
    Manages workload plans (JSON) that prescribe ingestion rates
    and workload profile shifts across simulation steps.
    """

    def __init__(self):
        self._lock = Lock()
        self._is_active = False
        self._plan_name = ""
        self._ingestion_schedule: List[Dict[str, Any]] = []
        self._workload_schedule: List[Dict[str, Any]] = []

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._is_active

    @property
    def plan_name(self) -> str:
        with self._lock:
            return self._plan_name

    def load_plan(self, json_data: Dict[str, Any]) -> str:
        """Load a v5 workload plan JSON."""
        try:
            with self._lock:
                self._plan_name = json_data.get('name', 'Unnamed Plan')
                self._ingestion_schedule = json_data.get('ingestion_schedule', [])
                self._workload_schedule = json_data.get('workload_schedule', [])
                self._is_active = True
                return (
                    f"✅ Loaded plan '{self._plan_name}': "
                    f"{len(self._ingestion_schedule)} ingestion phases, "
                    f"{len(self._workload_schedule)} workload phases"
                )
        except Exception as e:
            return f"❌ Failed to load plan: {e}"

    def get_ingestion_rate(self, step: int) -> Optional[int]:
        """Get prescribed ingestion rate for this step, or None."""
        with self._lock:
            if not self._is_active:
                return None
            for phase in self._ingestion_schedule:
                if phase['start'] <= step <= phase['end']:
                    return random.randint(phase['rows_min'], phase['rows_max'])
            return None

    def get_ingestion_description(self, step: int) -> Optional[str]:
        """Get human-readable description of the current ingestion phase."""
        with self._lock:
            if not self._is_active:
                return None
            for phase in self._ingestion_schedule:
                if phase['start'] <= step <= phase['end']:
                    return phase.get('description', f"{phase['rows_min']}-{phase['rows_max']} rows")
            return None

    def get_workload_profile(self, step: int) -> Optional[str]:
        """Get prescribed workload profile for this step, or None."""
        with self._lock:
            if not self._is_active:
                return None
            for phase in self._workload_schedule:
                if phase['start'] <= step <= phase['end']:
                    return phase['profile']
            return None

    def clear(self):
        with self._lock:
            self._is_active = False
            self._plan_name = ""
            self._ingestion_schedule = []
            self._workload_schedule = []


# ═══════════════════════════════════════════════════════════════
# REWARD CALCULATOR
# ═══════════════════════════════════════════════════════════════

class RewardCalculator:
    """
    Computes three reward signals per step.

    Compaction reward — file-level health (no latency — avoids attribution problems)
    Partition reward  — query-level health (pruning ratio, skew)
    Global reward     — overall system health (latency + file + pruning + util)
    """

    # Normalization ceilings (tuned for 30-300 rows/step ingestion scale)
    LATENCY_MAX        = 15000.0   # queries on 1000+ files can reach 10-15s
    FILE_COUNT_MAX     = 2000.0    # No_Maintenance can accumulate 2000+ files
    COMPACT_COST_MAX   = 1.1       # max compaction cost (C128)
    PARTITION_COST_MAX = 2.0       # max partition cost
    SKEW_MAX           = 2.0       # coefficient-of-variation cap

    # ── Compaction weights ──
    CW_FILES  = 0.25
    CW_UTIL   = 0.20
    CW_COST   = 0.10
    CW_TARGET = 0.45

    # ── Partition weights ──
    PW_PRUNING  = 0.55
    PW_DPRUNING = 0.25
    PW_SKEW     = 0.10
    PW_COST     = 0.10

    # ── Global weights ──
    GW_LATENCY = 0.30
    GW_FILES   = 0.20
    GW_PRUNING = 0.30
    GW_UTIL    = 0.20

    @classmethod
    def _compaction_target_bonus(cls, action: Action, rows_ingested: int) -> float:
        """Phase-aware target preference bonus.

        Intended behavior:
          • low ingest (<=50)     → prefer C32
          • medium ingest (51-149)→ prefer C64
          • burst ingest (>=150)  → prefer C128
        """
        if action not in (Action.COMPACT_32KB, Action.COMPACT_64KB, Action.COMPACT_128KB):
            return 0.0

        if rows_ingested <= 50:
            table = {
                Action.COMPACT_32KB: 1.00,
                Action.COMPACT_64KB: 0.25,
                Action.COMPACT_128KB: -0.50,
            }
        elif rows_ingested >= 150:
            table = {
                Action.COMPACT_32KB: -0.60,
                Action.COMPACT_64KB: 0.40,
                Action.COMPACT_128KB: 1.00,
            }
        else:
            table = {
                Action.COMPACT_32KB: 0.15,
                Action.COMPACT_64KB: 1.00,
                Action.COMPACT_128KB: 0.15,
            }
        return table.get(action, 0.0)

    @classmethod
    def compact_reward(
        cls,
        file_count: int,
        block_util: float,
        action_cost: float,
        action: Action,
        rows_ingested: int,
    ) -> float:
        f_norm = min(file_count / cls.FILE_COUNT_MAX, 1.0)
        u_norm = min(max(block_util, 0.0), 1.0)
        c_norm = action_cost / cls.COMPACT_COST_MAX if cls.COMPACT_COST_MAX > 0 else 0.0
        target_bonus = cls._compaction_target_bonus(action, rows_ingested)
        return (
            -cls.CW_FILES * f_norm
            + cls.CW_UTIL * u_norm
            - cls.CW_COST * c_norm
            + cls.CW_TARGET * target_bonus
        )

    @classmethod
    def partition_reward(
        cls,
        pruning_ratio: float,
        avg_pruning_prev: float,
        avg_pruning_now: float,
        partition_skew: float,
        action_cost: float,
    ) -> float:
        p_norm = min(max(pruning_ratio, 0.0), 1.0)
        dp = min(max(avg_pruning_now - avg_pruning_prev, -1.0), 1.0)
        s_norm = min(partition_skew / cls.SKEW_MAX, 1.0)
        c_norm = action_cost / cls.PARTITION_COST_MAX if cls.PARTITION_COST_MAX > 0 else 0.0
        return cls.PW_PRUNING * p_norm + cls.PW_DPRUNING * dp - cls.PW_SKEW * s_norm - cls.PW_COST * c_norm

    @classmethod
    def global_reward(
        cls,
        latency_ms: float,
        file_count: int,
        pruning_ratio: float,
        block_util: float,
    ) -> float:
        l_norm = min(max(latency_ms, 0.0) / cls.LATENCY_MAX, 1.0)
        f_norm = min(file_count / cls.FILE_COUNT_MAX, 1.0)
        p_norm = min(max(pruning_ratio, 0.0), 1.0)
        u_norm = min(max(block_util, 0.0), 1.0)
        return -cls.GW_LATENCY * l_norm - cls.GW_FILES * f_norm + cls.GW_PRUNING * p_norm + cls.GW_UTIL * u_norm

    @classmethod
    def error_reward(cls) -> Tuple[float, float, float]:
        """All-bad rewards on error."""
        return -1.0, -1.0, -1.0


# ═══════════════════════════════════════════════════════════════
# SPARK CONFIG
# ═══════════════════════════════════════════════════════════════

class SparkConfig:
    """Spark connection configuration for local research mode."""

    MASTER_URL     = "local[1]"
    APP_NAME       = "LakeGym-v5"
    CATALOG_NAME   = "iceberg"
    CATALOG_TYPE   = "rest"
    CATALOG_URI    = "http://rest:8181"
    WAREHOUSE_PATH = "s3://warehouse/"
    S3_ENDPOINT    = "http://minio:9000"
    S3_ACCESS_KEY  = "admin"
    S3_SECRET_KEY  = "password"
    S3_REGION      = "eu-central-1"
    DRIVER_MEMORY  = "2g"

    @classmethod
    def get_spark_configs(cls) -> Dict[str, str]:
        cat = cls.CATALOG_NAME
        return {
            "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            f"spark.sql.catalog.{cat}": "org.apache.iceberg.spark.SparkCatalog",
            f"spark.sql.catalog.{cat}.type": cls.CATALOG_TYPE,
            f"spark.sql.catalog.{cat}.uri": cls.CATALOG_URI,
            f"spark.sql.catalog.{cat}.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
            f"spark.sql.catalog.{cat}.warehouse": cls.WAREHOUSE_PATH,
            f"spark.sql.catalog.{cat}.s3.endpoint": cls.S3_ENDPOINT,
            f"spark.sql.catalog.{cat}.s3.path-style-access": "true",
            f"spark.sql.catalog.{cat}.s3.access-key-id": cls.S3_ACCESS_KEY,
            f"spark.sql.catalog.{cat}.s3.secret-access-key": cls.S3_SECRET_KEY,
            f"spark.sql.catalog.{cat}.s3.region": cls.S3_REGION,
            "spark.sql.defaultCatalog": cat,
            "spark.hadoop.fs.s3a.endpoint": cls.S3_ENDPOINT,
            "spark.hadoop.fs.s3a.access.key": cls.S3_ACCESS_KEY,
            "spark.hadoop.fs.s3a.secret.key": cls.S3_SECRET_KEY,
            "spark.hadoop.fs.s3a.endpoint.region": cls.S3_REGION,
            "spark.hadoop.fs.s3a.path.style.access": "true",
            "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
            "spark.sql.shuffle.partitions": "2",
            "spark.memory.fraction": "0.6",
            "spark.memory.storageFraction": "0.3",
            "spark.driver.memory": cls.DRIVER_MEMORY,
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.iceberg.distribution-mode": "none",
            "spark.sql.catalog.iceberg.cache-enabled": "false",
            "spark.sql.files.maxPartitionBytes": "8388608",
            "spark.sql.files.openCostInBytes": "1048576",
            "spark.eventLog.enabled": "false",
        }


# ═══════════════════════════════════════════════════════════════
# LAKE SIMULATOR
# ═══════════════════════════════════════════════════════════════

class LakeSimulator:
    """
    Core simulation engine.

    Manages one Iceberg table (`events`) and exposes:
      • run_step(action) → StepResult
      • get_observation() → dict (compatible with Observation.from_dict)
    """

    # ── Table ──
    CATALOG  = "iceberg"
    DATABASE = "default"
    TABLE    = "events"
    FULL_TABLE_NAME = f"{CATALOG}.{DATABASE}.{TABLE}"

    # ── Data generation ──
    PAYLOAD_SIZE    = 100           # 100 bytes → ~200 bytes/row → ~8 KB/file
    ROWS_PER_FILE   = 50            # 50 rows × 200 B ≈ 8 KB/file (well below C32 24KB min-file-size)
    LOW_LOAD_ROWS   = 30            # ~1 file of 5 KB
    HIGH_LOAD_ROWS  = 100           # ~2 files of 8 KB each
    NUM_SENSORS     = 10
    REGIONS         = ["US", "EU", "APAC"]
    REGION_WEIGHTS  = [0.50, 0.30, 0.20]
    EVENT_TYPES     = ["reading", "alert", "command"]
    EVENT_TYPE_WEIGHTS = [0.70, 0.20, 0.10]
    STATUS_OPTIONS  = ["OK", "WARN", "CRITICAL"]
    STATUS_WEIGHTS  = [0.85, 0.12, 0.03]
    TEMP_MIN, TEMP_MAX = -20.0, 50.0
    HUMID_MIN, HUMID_MAX = 0.0, 100.0

    # ── Cooldowns ──
    COMPACT_COOLDOWN   = 3
    PARTITION_COOLDOWN = 20

    # ── Query history window ──
    QUERY_HISTORY_SIZE = 20

    # ─────────────────────────────────────────────────────────
    # Init / Reset
    # ─────────────────────────────────────────────────────────

    def __init__(self):
        self._spark: Optional[SparkSession] = None
        self._state = SimulatorState()
        self._lock = Lock()
        self._history: deque = deque(maxlen=2000)
        self._cumulative_global: float = 0.0
        self._config = SimulationConfig()
        self._scenario = ScenarioManager()

        # Query history (rolling window of recent QueryType enums)
        self._query_history: deque = deque(maxlen=self.QUERY_HISTORY_SIZE)
        # Rolling pruning history for avg computation
        self._pruning_history: deque = deque(maxlen=self.QUERY_HISTORY_SIZE)
        self._prev_avg_pruning: float = 0.0

        # Payload template (reused for speed)
        self._payload = ''.join(
            random.choices(string.ascii_letters + string.digits, k=self.PAYLOAD_SIZE)
        )

    @property
    def state(self) -> SimulatorState:
        with self._lock:
            # Return a shallow copy
            s = self._state
            return SimulatorState(
                step_count=s.step_count, total_rows=s.total_rows,
                current_latency=s.current_latency, file_count=s.file_count,
                total_size_kb=s.total_size_kb, partition_strategy=s.partition_strategy,
                last_action=s.last_action, is_initialized=s.is_initialized,
                last_compact_step=s.last_compact_step, last_partition_step=s.last_partition_step,
                rows_at_partition_change=s.rows_at_partition_change,
                compacted_after_partition=s.compacted_after_partition,
                current_profile=s.current_profile, last_query_type=s.last_query_type,
            )

    @property
    def history(self) -> List[StepResult]:
        with self._lock:
            return list(self._history)

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def scenario_manager(self) -> ScenarioManager:
        return self._scenario

    # ─────────────────────────────────────────────────────────
    # Spark & Table management
    # ─────────────────────────────────────────────────────────

    def initialize(self) -> str:
        """Create Spark session and Iceberg table."""
        try:
            builder = SparkSession.builder.appName(SparkConfig.APP_NAME).master(SparkConfig.MASTER_URL)
            for k, v in SparkConfig.get_spark_configs().items():
                builder = builder.config(k, v)
            self._spark = builder.getOrCreate()
            self._spark.sparkContext.setLogLevel("ERROR")

            self._spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {self.CATALOG}.{self.DATABASE}")
            self._spark.sql(f"DROP TABLE IF EXISTS {self.FULL_TABLE_NAME}")
            self._spark.sql(f"""
                CREATE TABLE {self.FULL_TABLE_NAME} (
                    event_ts    TIMESTAMP,
                    sensor_id   STRING,
                    region      STRING,
                    event_type  STRING,
                    temperature FLOAT,
                    humidity    FLOAT,
                    status      STRING,
                    payload     STRING
                ) USING iceberg
            """)

            with self._lock:
                self._state = SimulatorState(is_initialized=True)
            return f"✅ Initialized: {self.FULL_TABLE_NAME}"
        except Exception as e:
            return f"❌ Init failed: {e}"

    def reset(self) -> str:
        """Drop and recreate the table, reset all state."""
        try:
            if not self._spark:
                return "❌ Spark not initialized"
            self._spark.catalog.clearCache()
            self._spark.sql(f"DROP TABLE IF EXISTS {self.FULL_TABLE_NAME}")
            self._spark.sql(f"""
                CREATE TABLE {self.FULL_TABLE_NAME} (
                    event_ts    TIMESTAMP,
                    sensor_id   STRING,
                    region      STRING,
                    event_type  STRING,
                    temperature FLOAT,
                    humidity    FLOAT,
                    status      STRING,
                    payload     STRING
                ) USING iceberg
            """)
            with self._lock:
                self._state = SimulatorState(is_initialized=True)
                self._history.clear()
                self._cumulative_global = 0.0
                self._query_history.clear()
                self._pruning_history.clear()
                self._prev_avg_pruning = 0.0
            return "✅ Simulation reset"
        except Exception as e:
            return f"❌ Reset failed: {e}"

    # ─────────────────────────────────────────────────────────
    # Ingestion
    # ─────────────────────────────────────────────────────────

    def ingest_micro_batch(self) -> Tuple[int, float, str]:
        """
        Ingest rows into the events table, creating multiple small files.

        Returns (rows_ingested, elapsed_seconds, message).
        """
        if not self._spark:
            return 0, 0.0, "❌ Spark not initialized"

        try:
            start = time.time()

            # Determine row count
            step = self._state.step_count
            plan_rate = self._scenario.get_ingestion_rate(step)
            if plan_rate is not None:
                num_rows = plan_rate
                label = "PLAN"
            elif self._config.manual_override_active:
                num_rows = self._config.ingestion_rate_rows
                label = "MANUAL"
            else:
                num_rows = self.HIGH_LOAD_ROWS if random.random() > 0.6 else self.LOW_LOAD_ROWS
                label = "HIGH" if num_rows == self.HIGH_LOAD_ROWS else "LOW"

            num_files = max(1, num_rows // self.ROWS_PER_FILE)

            now = datetime.now()
            data = {
                'event_ts':    [now - timedelta(seconds=random.randint(0, 300)) for _ in range(num_rows)],
                'sensor_id':   [f"S-{random.randint(1, self.NUM_SENSORS):03d}" for _ in range(num_rows)],
                'region':      random.choices(self.REGIONS, weights=self.REGION_WEIGHTS, k=num_rows),
                'event_type':  random.choices(self.EVENT_TYPES, weights=self.EVENT_TYPE_WEIGHTS, k=num_rows),
                'temperature': [random.uniform(self.TEMP_MIN, self.TEMP_MAX) for _ in range(num_rows)],
                'humidity':    [random.uniform(self.HUMID_MIN, self.HUMID_MAX) for _ in range(num_rows)],
                'status':      random.choices(self.STATUS_OPTIONS, weights=self.STATUS_WEIGHTS, k=num_rows),
                'payload':     [''.join(random.choices(string.ascii_letters + string.digits, k=self.PAYLOAD_SIZE)) for _ in range(num_rows)],
            }
            df = self._spark.createDataFrame(pd.DataFrame(data))
            df.repartition(num_files).writeTo(self.FULL_TABLE_NAME).append()

            elapsed = time.time() - start
            with self._lock:
                self._state.total_rows += num_rows
            return num_rows, elapsed, f"📥 {num_rows} rows → {num_files} files ({label})"
        except Exception as e:
            return 0, 0.0, f"❌ Ingest failed: {e}"

    # ─────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────

    def collect_metrics(self) -> Tuple[int, float, str]:
        """Collect file count and total size."""
        if not self._spark:
            return 0, 0.0, "❌ not init"
        try:
            row = self._spark.sql(f"""
                SELECT COUNT(*) AS fc, COALESCE(SUM(file_size_in_bytes),0) AS tb
                FROM {self.FULL_TABLE_NAME}.files
            """).collect()[0]
            fc = int(row['fc'])
            sz = float(row['tb']) / 1024.0
            with self._lock:
                self._state.file_count = fc
                self._state.total_size_kb = sz
            return fc, sz, f"📊 {fc} files, {sz:.1f} KB"
        except Exception as e:
            return 0, 0.0, f"❌ Metrics: {e}"

    def collect_advanced_metrics(self) -> Dict[str, float]:
        """Block utilization, file size skew, partition skew."""
        defaults = {'block_utilization': 0.0, 'file_size_skew_kb': 0.0, 'partition_skew': 0.0}
        if not self._spark:
            return defaults
        try:
            row = self._spark.sql(f"""
                SELECT COUNT(*) AS fc,
                       COALESCE(AVG(file_size_in_bytes),0) AS avg_b,
                       COALESCE(STDDEV(file_size_in_bytes),0) AS std_b
                FROM {self.FULL_TABLE_NAME}.files
            """).collect()[0]
            avg_b = float(row['avg_b'] or 0)
            std_b = float(row['std_b'] or 0)
            optimal_b = OPTIMAL_FILE_SIZE_KB * 1024
            block_util = avg_b / optimal_b if optimal_b > 0 else 0.0

            # Partition skew (coefficient of variation)
            partition_skew = 0.0
            try:
                ps = self._spark.sql(f"""
                    SELECT record_count FROM {self.FULL_TABLE_NAME}.partitions
                """).collect()
                if len(ps) > 1:
                    counts = [float(r['record_count']) for r in ps]
                    mean_c = np.mean(counts)
                    std_c = np.std(counts)
                    partition_skew = float(std_c / mean_c) if mean_c > 0 else 0.0
            except Exception:
                pass

            return {
                'block_utilization': block_util,
                'file_size_skew_kb': std_b / 1024.0,
                'partition_skew': partition_skew,
            }
        except Exception as e:
            print(f"⚠️ Advanced metrics: {e}")
            return defaults

    # ─────────────────────────────────────────────────────────
    # Query execution
    # ─────────────────────────────────────────────────────────

    def _pick_query_type(self) -> QueryType:
        """Sample a query type from the current workload profile."""
        with self._lock:
            profile_name = self._state.current_profile
        profile = WORKLOAD_PROFILES.get(profile_name, WORKLOAD_PROFILES["mixed"])
        types = list(profile.keys())
        weights = list(profile.values())
        return random.choices(types, weights=weights, k=1)[0]

    def _build_query(self, qtype: QueryType) -> Tuple[str, str]:
        """Build SQL string for the given query type. Returns (sql, label)."""
        t = self.FULL_TABLE_NAME
        if qtype == QueryType.TIME_RANGE:
            sql = f"""
                SELECT hour(event_ts) AS hr, avg(temperature) AS avg_temp, avg(humidity) AS avg_hum
                FROM {t}
                WHERE event_ts > current_timestamp() - INTERVAL 1 DAY
                GROUP BY 1 ORDER BY 1
            """
            return sql, "TIME_RANGE"
        elif qtype == QueryType.REGION_FILTER:
            region = random.choice(self.REGIONS)
            sql = f"""
                SELECT region, count(*) AS cnt, avg(temperature) AS avg_temp
                FROM {t}
                WHERE region = '{region}'
                GROUP BY 1
            """
            return sql, f"REGION_FILTER({region})"
        elif qtype == QueryType.SENSOR_LOOKUP:
            sid = f"S-{random.randint(1, self.NUM_SENSORS):03d}"
            sql = f"""
                SELECT sensor_id, event_ts, temperature, humidity
                FROM {t}
                WHERE sensor_id = '{sid}' AND event_ts > current_timestamp() - INTERVAL 1 HOUR
                ORDER BY event_ts DESC
            """
            return sql, f"SENSOR_LOOKUP({sid})"
        elif qtype == QueryType.TYPE_FILTER:
            etype = random.choice(self.EVENT_TYPES)
            sql = f"""
                SELECT event_type, avg(humidity) AS avg_hum, max(temperature) AS max_temp, count(*) AS cnt
                FROM {t}
                WHERE event_type = '{etype}'
                GROUP BY 1
            """
            return sql, f"TYPE_FILTER({etype})"
        else:  # FULL_SCAN
            sql = f"""
                SELECT count(*) AS total, avg(temperature) AS avg_temp, stddev(humidity) AS std_hum
                FROM {t}
            """
            return sql, "FULL_SCAN"

    def _compute_pruning_ratio(self, qtype: QueryType) -> float:
        """
        Deterministic pruning ratio based on query type + partition strategy
        + how much data was written after the partition was set.
        """
        with self._lock:
            ps = self._state.partition_strategy
            compacted = self._state.compacted_after_partition
            total_rows = self._state.total_rows
            rows_at_change = self._state.rows_at_partition_change

        if ps == PartitionStrategy.UNPARTITIONED:
            return 0.0

        # Sensitivity-analysis scaling (revision Phase 3): multiply all non-zero
        # pruning values by a constant, clipped to [0, 1], to test whether the
        # RL-vs-heuristic gap survives a weaker (or stronger) pruning benefit.
        # Defaults to 1.0, so behaviour is unchanged unless explicitly set.
        base = _BASE_PRUNING.get((qtype, ps), 0.0)
        if base == 0.0:
            return 0.0
        base = min(max(base * PRUNING_SCALE, 0.0), 1.0)

        # Coverage: fraction of data in the partitioned layout
        if compacted:
            coverage = 1.0
        else:
            rows_after = total_rows - rows_at_change
            coverage = rows_after / max(total_rows, 1)

        return base * coverage

    def measure_performance(self) -> Tuple[float, QueryType, str]:
        """
        Pick a query from the current workload profile, execute it, measure latency.

        Returns (latency_ms, query_type, label).
        """
        if not self._spark:
            return -1.0, QueryType.FULL_SCAN, "❌ not init"
        try:
            self._spark.catalog.clearCache()

            # Determine workload profile for this step
            step = self._state.step_count
            plan_profile = self._scenario.get_workload_profile(step)
            if plan_profile and plan_profile in WORKLOAD_PROFILES:
                with self._lock:
                    self._state.current_profile = plan_profile
            elif self._config.manual_profile and self._config.manual_profile in WORKLOAD_PROFILES:
                with self._lock:
                    self._state.current_profile = self._config.manual_profile

            # Pick query type
            if self._config.manual_override_active and self._config.manual_query_type:
                try:
                    qtype = QueryType[self._config.manual_query_type]
                except KeyError:
                    qtype = self._pick_query_type()
            else:
                qtype = self._pick_query_type()

            sql, label = self._build_query(qtype)

            start = time.time()
            result = self._spark.sql(sql).collect()
            latency_ms = (time.time() - start) * 1000.0

            if not result:
                return 0.0, qtype, f"⚠️ {label}: no data"

            with self._lock:
                self._state.current_latency = latency_ms
                self._state.last_query_type = qtype

            # Track query history
            self._query_history.append(qtype)

            return latency_ms, qtype, f"⏱️ {label}: {latency_ms:.0f} ms"
        except Exception as e:
            return -1.0, QueryType.FULL_SCAN, f"❌ Query: {e}"

    # ─────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────

    def execute_action(self, action: Action) -> Tuple[float, str, bool]:
        """
        Execute the given action. Returns (cost, message, success).
        """
        if not self._spark:
            return 0.0, "❌ not init", False

        cost = ACTION_COSTS.get(action, 0.0)

        with self._lock:
            current_ps = self._state.partition_strategy
            step = self._state.step_count

        # ── Skip redundant partition actions ──
        ps_map = {
            Action.PARTITION_HOUR: PartitionStrategy.HOUR,
            Action.PARTITION_REGION: PartitionStrategy.REGION,
            Action.PARTITION_EVENT_TYPE: PartitionStrategy.EVENT_TYPE,
        }
        if action in ps_map and current_ps == ps_map[action]:
            return cost + 3.0, f"⏭️ Already {current_ps.name} partitioned", False

        if action == Action.REMOVE_PARTITION and current_ps == PartitionStrategy.UNPARTITIONED:
            return cost, "⏭️ Already unpartitioned", False

        try:
            if action == Action.NOOP:
                return 0.0, "💤 NOOP", True

            elif action in (Action.COMPACT_32KB, Action.COMPACT_64KB, Action.COMPACT_128KB):
                target_map = {Action.COMPACT_32KB: 32, Action.COMPACT_64KB: 64, Action.COMPACT_128KB: 128}
                target_kb = target_map[action]
                ok = self._run_compaction(target_kb)
                with self._lock:
                    self._state.last_compact_step = step
                    # Compaction rewrites data into current partition layout
                    if self._state.partition_strategy != PartitionStrategy.UNPARTITIONED:
                        self._state.compacted_after_partition = True
                return cost, f"🗜️ Compact({target_kb}KB)", ok

            elif action in (Action.PARTITION_HOUR, Action.PARTITION_REGION, Action.PARTITION_EVENT_TYPE):
                target_ps = ps_map[action]
                ok = self._set_partition_strategy(target_ps)
                if ok:
                    with self._lock:
                        self._state.last_partition_step = step
                return cost, f"📂 Partition → {target_ps.name}", ok

            elif action == Action.REMOVE_PARTITION:
                ok = self._remove_partition()
                if ok:
                    with self._lock:
                        self._state.last_partition_step = step
                return cost, "🗑️ Partition removed", ok

            return cost, f"❓ Unknown action: {action}", False
        except Exception as e:
            return cost, f"❌ Action failed: {e}", False

    def _run_compaction(self, target_kb: int = 64) -> bool:
        try:
            fc = self._spark.sql(
                f"SELECT COUNT(*) AS c FROM {self.FULL_TABLE_NAME}.files"
            ).collect()[0]['c']
            if fc < 2:
                return True  # nothing to compact

            target_bytes = target_kb * 1024
            self._spark.sql(f"""
                CALL {self.CATALOG}.system.rewrite_data_files(
                    table => '{self.DATABASE}.{self.TABLE}',
                    strategy => 'binpack',
                    options => map(
                        'target-file-size-bytes', '{target_bytes}',
                        'min-input-files', '2'
                    )
                )
            """)
            self._spark.catalog.clearCache()
            return True
        except Exception as e:
            print(f"❌ Compaction failed: {e}")
            return False

    def _set_partition_strategy(self, target: PartitionStrategy) -> bool:
        """Apply a new partition strategy via Iceberg partition evolution."""
        with self._lock:
            current = self._state.partition_strategy

        if target == current:
            return True

        try:
            # Remove old partition field if any
            self._drop_current_partition(current)

            # Add new partition field
            if target == PartitionStrategy.HOUR:
                self._spark.sql(f"ALTER TABLE {self.FULL_TABLE_NAME} ADD PARTITION FIELD hours(event_ts)")
            elif target == PartitionStrategy.REGION:
                self._spark.sql(f"ALTER TABLE {self.FULL_TABLE_NAME} ADD PARTITION FIELD identity(region)")
            elif target == PartitionStrategy.EVENT_TYPE:
                self._spark.sql(f"ALTER TABLE {self.FULL_TABLE_NAME} ADD PARTITION FIELD identity(event_type)")

            with self._lock:
                self._state.partition_strategy = target
                self._state.rows_at_partition_change = self._state.total_rows
                self._state.compacted_after_partition = False  # need compaction to rewrite old data

            print(f"✅ Partition: {current.name} → {target.name}")
            return True
        except Exception as e:
            print(f"❌ Partition change failed: {e}")
            return False

    def _remove_partition(self) -> bool:
        """Remove the current partition field."""
        with self._lock:
            current = self._state.partition_strategy
        if current == PartitionStrategy.UNPARTITIONED:
            return True
        try:
            self._drop_current_partition(current)
            with self._lock:
                self._state.partition_strategy = PartitionStrategy.UNPARTITIONED
                self._state.rows_at_partition_change = self._state.total_rows
                self._state.compacted_after_partition = True
            return True
        except Exception as e:
            print(f"❌ Remove partition failed: {e}")
            return False

    def _drop_current_partition(self, ps: PartitionStrategy):
        """Drop the partition field for the given strategy."""
        if ps == PartitionStrategy.HOUR:
            self._spark.sql(f"ALTER TABLE {self.FULL_TABLE_NAME} DROP PARTITION FIELD hours(event_ts)")
        elif ps == PartitionStrategy.REGION:
            self._spark.sql(f"ALTER TABLE {self.FULL_TABLE_NAME} DROP PARTITION FIELD identity(region)")
        elif ps == PartitionStrategy.EVENT_TYPE:
            self._spark.sql(f"ALTER TABLE {self.FULL_TABLE_NAME} DROP PARTITION FIELD identity(event_type)")

    # ─────────────────────────────────────────────────────────
    # Query history helpers
    # ─────────────────────────────────────────────────────────

    def _query_histogram(self) -> Dict[str, float]:
        """Compute fraction of each query type in recent history."""
        n = len(self._query_history)
        if n == 0:
            return {qt.name.lower(): 0.0 for qt in QueryType}
        counts = {qt: 0 for qt in QueryType}
        for qt in self._query_history:
            counts[qt] += 1
        return {
            'time_range':    counts[QueryType.TIME_RANGE] / n,
            'region_filter': counts[QueryType.REGION_FILTER] / n,
            'sensor_lookup': counts[QueryType.SENSOR_LOOKUP] / n,
            'type_filter':   counts[QueryType.TYPE_FILTER] / n,
            'full_scan':     counts[QueryType.FULL_SCAN] / n,
        }

    # ─────────────────────────────────────────────────────────
    # Full Step
    # ─────────────────────────────────────────────────────────

    def run_step(self, action: Action = Action.NOOP) -> StepResult:
        """
        Execute one full simulation step:
          1. Execute action
          2. Ingest data (paused if maintenance ran)
          3. Collect metrics
          4. Run query & measure latency
          5. Compute rewards
          6. Record result
        """
        with self._lock:
            self._state.step_count += 1
            step_num = self._state.step_count
            self._state.last_action = action

        timestamp = datetime.now().isoformat()
        error_msg = None
        rows = 0
        ingestion_rate = 0.0

        try:
            # 1. Ingest first (writes are never blocked — Iceberg uses optimistic concurrency)
            rows, elapsed, _ = self.ingest_micro_batch()
            ingestion_rate = rows / elapsed if elapsed > 0 else 0.0

            # 2. Execute action (compaction / partition change / noop)
            action_cost, action_msg, action_success = self.execute_action(action)

            # 3. Metrics
            file_count, total_size_kb, _ = self.collect_metrics()
            adv = self.collect_advanced_metrics()
            block_util = adv['block_utilization']
            skew_kb = adv['file_size_skew_kb']
            partition_skew = adv['partition_skew']

            # 4. Query
            latency_ms, qtype, _ = self.measure_performance()
            pruning_ratio = self._compute_pruning_ratio(qtype)

            # Update pruning history
            self._pruning_history.append(pruning_ratio)
            avg_pruning_now = float(np.mean(self._pruning_history)) if self._pruning_history else 0.0

            # 5. Rewards
            r_compact = RewardCalculator.compact_reward(
                file_count=file_count,
                block_util=block_util,
                action_cost=action_cost,
                action=action,
                rows_ingested=rows,
            )
            r_partition = RewardCalculator.partition_reward(
                pruning_ratio, self._prev_avg_pruning, avg_pruning_now, partition_skew, action_cost
            )
            r_global = RewardCalculator.global_reward(latency_ms, file_count, pruning_ratio, block_util)

            self._prev_avg_pruning = avg_pruning_now
            self._cumulative_global += r_global

            partition_count = self._get_partition_count()

        except Exception as e:
            error_msg = str(e)
            traceback.print_exc()
            latency_ms, qtype = -1.0, QueryType.FULL_SCAN
            r_compact, r_partition, r_global = RewardCalculator.error_reward()
            action_success = False
            file_count, total_size_kb = 0, 0.0
            block_util, skew_kb, partition_skew = 0.0, 0.0, 0.0
            pruning_ratio, partition_count = 0.0, 0
            action_cost = 0.0

        # 6. Build result
        with self._lock:
            ps_name = self._state.partition_strategy.name
            profile = self._state.current_profile

        result = StepResult(
            step=step_num,
            timestamp=timestamp,
            rows_ingested=rows,
            total_rows=self._state.total_rows,
            latency_ms=latency_ms,
            file_count=file_count,
            total_size_kb=total_size_kb,
            partition_strategy=ps_name,
            partition_count=partition_count,
            action_taken=action.name,
            action_success=action_success,
            ingestion_rate_rows_per_sec=ingestion_rate,
            block_utilization=block_util,
            file_size_skew_kb=skew_kb,
            partition_skew=partition_skew,
            partition_pruning_ratio=pruning_ratio,
            compact_reward=r_compact,
            partition_reward=r_partition,
            global_reward=r_global,
            cumulative_global_reward=self._cumulative_global,
            query_type=QUERY_TYPE_NAMES.get(qtype, "UNKNOWN"),
            workload_profile=profile,
            error=error_msg,
        )

        with self._lock:
            self._history.append(result)
        return result

    # ─────────────────────────────────────────────────────────
    # Observation (for agents / policies)
    # ─────────────────────────────────────────────────────────

    def get_observation(self) -> Dict[str, Any]:
        """
        Build the full observation dictionary.
        Compatible with Observation.from_dict().
        """
        hist = self._query_histogram()

        with self._lock:
            last = self._history[-1] if self._history else None
            s = self._state

            rows_ingested = last.rows_ingested if last else 0
            ingestion_rate = last.ingestion_rate_rows_per_sec if last else 0.0
            latency_ms = max(s.current_latency, 0.0)
            block_util = last.block_utilization if last else 0.0
            skew_kb = last.file_size_skew_kb if last else 0.0
            pruning = last.partition_pruning_ratio if last else 0.0

            steps_since_compact = s.step_count - s.last_compact_step
            steps_since_partition = s.step_count - s.last_partition_step

        avg_pruning = float(np.mean(self._pruning_history)) if self._pruning_history else 0.0

        return {
            'rows_ingested': rows_ingested,
            'ingestion_rate_rows_per_sec': ingestion_rate,
            'latency_ms': latency_ms,
            'file_count': s.file_count,
            'block_utilization': block_util,
            'total_size_kb': s.total_size_kb,
            'file_size_skew_kb': skew_kb,
            'partition_strategy': int(s.partition_strategy),
            'steps_since_compact': steps_since_compact,
            'steps_since_partition_change': steps_since_partition,
            'partition_pruning_ratio': pruning,
            'avg_pruning_ratio': avg_pruning,
            'query_hist_time_range': hist['time_range'],
            'query_hist_region_filter': hist['region_filter'],
            'query_hist_sensor_lookup': hist['sensor_lookup'],
            'query_hist_type_filter': hist['type_filter'],
            'query_hist_full_scan': hist['full_scan'],
        }

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _get_partition_count(self) -> int:
        if not self._spark:
            return 0
        try:
            r = self._spark.sql(f"""
                SELECT COUNT(DISTINCT partition) AS c FROM {self.FULL_TABLE_NAME}.partitions
            """).collect()[0]['c']
            return int(r) if r else 0
        except Exception:
            return 0

    def set_ingestion_rate(self, rows: int):
        self._config.ingestion_rate_rows = max(1, min(100, rows))

    def set_manual_override(self, enabled: bool):
        self._config.manual_override_active = enabled

    def export_history_to_csv(self, filepath: str) -> str:
        with self._lock:
            if not self._history:
                return "No history"
            records = [r.to_dict() for r in self._history]
        pd.DataFrame(records).to_csv(filepath, index=False)
        return f"✅ Exported {len(records)} steps to {filepath}"

    def get_summary_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._history:
                return {}
            df = pd.DataFrame([r.to_dict() for r in self._history])
        return {
            'total_steps': len(df),
            'avg_latency': round(df['latency_ms'].mean(), 2),
            'avg_global_reward': round(df['global_reward'].mean(), 4),
            'total_global_reward': round(df['global_reward'].sum(), 4),
        }
