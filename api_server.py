#!/usr/bin/env python3
"""FedFOD API Server — FastAPI REST + WebSocket Bridge
======================================================

Bridges the FedFOD federated learning backend (Flower gRPC) to a Flutter
frontend via a JSON REST API and real-time WebSocket streams.

Endpoints
---------
REST:
    GET  /api/status                     → server status & uptime
    GET  /api/config/global              → global_config.yaml as JSON
    PUT  /api/config/global              → update global_config.yaml
    GET  /api/config/airports            → list all airport configs
    GET  /api/config/airports/{id}       → single airport config
    GET  /api/metrics/report             → simulation_report.json
    GET  /api/metrics/rounds             → per-round metrics (JSONL)
    GET  /api/logs                       → last N log lines
    POST /api/training/start             → launch FL server + clients
    POST /api/training/stop              → kill all training processes
    GET  /api/training/state             → current training state

WebSocket:
    WS   /ws/metrics                     → stream round metrics live
    WS   /ws/logs                        → stream log lines live

Run:
    python api_server.py
    # → http://0.0.0.0:8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import datetime
import json
import logging
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64
import zipfile
import shutil
import cv2
import numpy as np
import torch
from prometheus_client import make_asgi_app, Gauge

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
PROM_CURRENT_ROUND = Gauge("fedfod_current_round", "Current federated learning round")
PROM_ACTIVE_CLIENTS = Gauge("fedfod_active_clients", "Number of active clients in the current round")
PROM_GLOBAL_MAP50 = Gauge("fedfod_global_map50", "Global model mAP@50 convergence")
PROM_GLOBAL_LOSS = Gauge("fedfod_global_loss", "Global training or evaluation loss")
PROM_GINI_COEFFICIENT = Gauge("fedfod_gini_coefficient", "Gini coefficient fairness rating")
PROM_CONNECTED_CLIENTS = Gauge("fedfod_connected_clients_count", "Number of connected clients to gRPC server")
PROM_COMMUNICATION_MB = Gauge("fedfod_communication_mb", "Total communication volume in MB")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
GLOBAL_CONFIG_PATH = CONFIG_DIR / "global_config.yaml"
AIRPORT_CONFIGS_DIR = CONFIG_DIR / "airport_configs"
LOGS_DIR = BASE_DIR / "logs"
ROUND_METRICS_PATH = LOGS_DIR / "round_metrics.jsonl"
SIMULATION_REPORT_PATH = LOGS_DIR / "simulation_report.json"
SERVER_RUNNER = BASE_DIR / "server_runner.py"
CLIENT_RUNNER = BASE_DIR / "client_runner.py"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("fedfod.api_server")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class TrainingPhase(str, Enum):
    """Possible phases of the training lifecycle."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class ClientSpec(BaseModel):
    """Specification for a single FL client to launch."""
    id: str = Field(..., description="Client identifier, e.g. '0', 'A'")
    airport_config: str = Field(
        ...,
        description="Relative path to airport YAML, e.g. 'config/airport_configs/airport_A.yaml'",
    )


class TrainingStartRequest(BaseModel):
    """Body for POST /api/training/start."""
    rounds: int = Field(default=90, ge=1, description="Number of FL rounds")
    min_clients: int = Field(default=2, ge=1, description="Min clients per round")
    port: int = Field(default=8080, ge=1024, le=65535, description="Flower gRPC port")
    dummy_model: bool = Field(default=False, description="Use DummyFODModel for testing")
    backbone: str = Field(default="yolov8n", description="Model backbone to use")
    clients: List[ClientSpec] = Field(
        default_factory=list,
        description="List of clients to launch. If empty, defaults to 3 dummy clients.",
    )


class StatusResponse(BaseModel):
    """Response for GET /api/status."""
    status: str
    uptime: float
    training_active: bool
    current_round: int
    connected_clients: int


class TrainingStateResponse(BaseModel):
    """Response for GET /api/training/state."""
    phase: TrainingPhase
    current_round: int
    total_rounds: int
    connected_clients: int
    start_time: Optional[str]
    elapsed_seconds: float
    server_pid: Optional[int]
    client_pids: List[int]
    last_error: Optional[str]


# ---------------------------------------------------------------------------
# Training State Singleton
# ---------------------------------------------------------------------------


class TrainingState:
    """Mutable singleton holding all training-related runtime state.

    This object is shared across request handlers and background tasks.
    All mutations must be protected by the asyncio lock when accessed from
    async code to prevent data races.
    """

    def __init__(self) -> None:
        self.phase: TrainingPhase = TrainingPhase.IDLE
        self.current_round: int = 0
        self.total_rounds: int = 0
        self.connected_clients: int = 0
        self.start_time: Optional[float] = None
        self.server_process: Optional[subprocess.Popen] = None
        self.client_processes: List[subprocess.Popen] = []
        self.monitor_task: Optional[asyncio.Task] = None
        self.last_error: Optional[str] = None
        self.lock = asyncio.Lock()

    # -- Convenience helpers ------------------------------------------------

    @property
    def server_pid(self) -> Optional[int]:
        if self.server_process and self.server_process.poll() is None:
            return self.server_process.pid
        return None

    @property
    def client_pids(self) -> List[int]:
        return [
            p.pid for p in self.client_processes if p.poll() is None
        ]

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def reset(self) -> None:
        """Reset state to idle defaults (processes must already be dead)."""
        self.phase = TrainingPhase.IDLE
        self.current_round = 0
        self.total_rounds = 0
        self.connected_clients = 0
        self.start_time = None
        self.server_process = None
        self.client_processes = []
        self.monitor_task = None
        self.last_error = None

    def to_response(self) -> TrainingStateResponse:
        return TrainingStateResponse(
            phase=self.phase,
            current_round=self.current_round,
            total_rounds=self.total_rounds,
            connected_clients=self.connected_clients,
            start_time=(
                datetime.datetime.fromtimestamp(self.start_time).isoformat()
                if self.start_time
                else None
            ),
            elapsed_seconds=round(self.elapsed, 2),
            server_pid=self.server_pid,
            client_pids=self.client_pids,
            last_error=self.last_error,
        )


# Global instances
_state = TrainingState()
_server_start_time = time.time()

# WebSocket broadcast channels — each is a set of connected queues
_metrics_subscribers: List[asyncio.Queue] = []
_logs_subscribers: List[asyncio.Queue] = []

# Global caches for lazy-loaded detector instances
_cached_detectors = {}
_cached_clip_detector = None


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: pathlib.Path) -> dict:
    """Read a YAML file and return its contents as a dict."""
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _save_yaml(path: pathlib.Path, data: dict) -> None:
    """Write a dict to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _collect_log_lines(n: int = 200) -> List[str]:
    """Return the last *n* lines across all ``*.log`` files in LOGS_DIR,
    sorted chronologically by the timestamp prefix if present."""
    lines: List[str] = []
    if not LOGS_DIR.is_dir():
        return lines
    for log_file in sorted(LOGS_DIR.glob("*.log")):
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            lines.extend(text.splitlines())
        except OSError:
            continue
    # Return last N lines (they are already chronological within each file)
    return lines[-n:]


# ---------------------------------------------------------------------------
# Round-metric regex (matches server_runner / simulation log output)
# ---------------------------------------------------------------------------

_ROUND_RE = re.compile(
    r"Round\s+(\d+)\s*(?:\||—|-|\ufffd)\s*mAP@50=([\d.]+)\s+"
    r"FAR/hr=([\d.]+)\s+comm=([\d.]+)MB\s+time=([\d.]+)s"
)

# New gRPC server format:
# "Round 1 — aggregated: 1 clients, loss=0.0000, mAP50=0.0000, mAP50-95=0.0000, Gini=0.0000 (valid=True)"
_ROUND_GRPC_RE = re.compile(
    r"Round\s+(\d+)\s*(?:\||—|-|\ufffd)\s*aggregated:\s*\d+\s*clients,\s*loss=([\d.]+),\s*mAP50=([\d.]+),\s*mAP50-95=([\d.]+),\s*Gini=([\d.]+)"
)



def _parse_round_line(line: str) -> Optional[Dict[str, Any]]:
    """Try to extract round metrics from a log line.

    Supports both manual simulation format and gRPC aggregator format.
    """
    m = _ROUND_RE.search(line)
    if m:
        return {
            "round": int(m.group(1)),
            "mAP50": float(m.group(2)),
            "FAR_per_hr": float(m.group(3)),
            "comm_MB": float(m.group(4)),
            "time_s": float(m.group(5)),
            "train_loss": 0.0,
            "eval_loss": 0.0,
            "gini": 0.0,
            "num_clients": 1,
        }

    m2 = _ROUND_GRPC_RE.search(line)
    if m2:
        return {
            "round": int(m2.group(1)),
            "train_loss": float(m2.group(2)),
            "mAP50": float(m2.group(3)),
            "mAP50_95": float(m2.group(4)),
            "gini": float(m2.group(5)),
            "FAR_per_hr": 0.0,
            "comm_MB": 0.0,
            "time_s": 1.0,
            "num_clients": 1,
        }

    return None


# ---------------------------------------------------------------------------
# Image weather analysis helper (Auto-Detect Weather)
# ---------------------------------------------------------------------------

def _analyze_image_weather(frame: np.ndarray) -> Dict[str, float]:
    """Automatically analyze image contrast and luminance to estimate weather context."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean()) / 255.0
    std_val = float(gray.std()) / 255.0
    
    # 1. Hour (Estimate based on average luminance)
    if mean_val < 0.15:
        hour = 2.0  # Night
    elif mean_val < 0.3:
        hour = 6.0  # Dawn/Dusk
    elif mean_val > 0.7:
        hour = 12.0 # Bright sun
    else:
        hour = 14.0 # Afternoon
        
    # 2. Fog Probability (Low contrast -> Foggy)
    if std_val < 0.08:
        fog_prob = 0.8
    elif std_val < 0.12:
        fog_prob = 0.4
    else:
        fog_prob = 0.0
        
    # 3. Glare Probability (Fraction of highly saturated white pixels)
    white_pixels = float((gray > 240).sum()) / gray.size
    if white_pixels > 0.05:
        glare_prob = 0.6
    elif white_pixels > 0.02:
        glare_prob = 0.3
    else:
        glare_prob = 0.05
        
    # 4. Rain Probability (Estimate based on reflections/glare and low contrast)
    if glare_prob > 0.4 and std_val > 0.15:
        rain_prob = 0.5  # Wet tarmac reflections
    else:
        rain_prob = 0.0
        
    return {
        "rain_prob": rain_prob,
        "fog_prob": fog_prob,
        "glare_prob": glare_prob,
        "hour": hour,
        "luminance_mean": mean_val,
        "luminance_std": std_val
    }


# ---------------------------------------------------------------------------
# WebSocket broadcast helper
# ---------------------------------------------------------------------------

async def _broadcast(subscribers: List[asyncio.Queue], message: str) -> None:
    """Push *message* to every subscriber queue; drop dead queues."""
    dead: List[asyncio.Queue] = []
    for q in subscribers:
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            subscribers.remove(q)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Background: monitor server stdout for round completions
# ---------------------------------------------------------------------------

async def _monitor_server_output(proc: subprocess.Popen) -> None:
    """Monitor server_runner by tailing logs/server.log.

    Reads new lines, parses round metrics, writes them to round_metrics.jsonl,
    updates training state, and broadcasts to WebSocket subscribers.
    """
    global _state
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    server_log_path = LOGS_DIR / "server.log"

    # Wait for the file to exist
    for _ in range(20):
        if server_log_path.is_file():
            break
        await asyncio.sleep(0.2)

    if not server_log_path.is_file():
        logger.error("server.log was not created. Monitor exiting.")
        async with _state.lock:
            _state.phase = TrainingPhase.FAILED
            _state.last_error = "server.log was not created."
        return

    offset = 0
    try:
        while True:
            # Check if process exited
            retcode = proc.poll()

            # Read any new data from the log file
            try:
                with open(server_log_path, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offset)
                    new_data = fh.read()
                    if new_data:
                        offset = fh.tell()
                        for line in new_data.splitlines():
                            line = line.rstrip("\n\r")
                            if not line:
                                continue

                            # Broadcast raw line to /ws/logs subscribers
                            await _broadcast(_logs_subscribers, line)

                            # Try to parse round metrics
                            metrics = _parse_round_line(line)
                            if metrics:
                                async with _state.lock:
                                    _state.current_round = metrics["round"]

                                # Update Prometheus Metrics
                                PROM_CURRENT_ROUND.set(metrics.get("round", 0))
                                PROM_GLOBAL_MAP50.set(metrics.get("mAP50", 0.0))
                                PROM_GLOBAL_LOSS.set(metrics.get("train_loss", metrics.get("eval_loss", 0.0)))
                                PROM_GINI_COEFFICIENT.set(metrics.get("gini", 0.0))
                                PROM_COMMUNICATION_MB.set(metrics.get("comm_MB", 0.0))
                                PROM_ACTIVE_CLIENTS.set(metrics.get("num_clients", 1))

                                # Append to JSONL file
                                try:
                                    with open(ROUND_METRICS_PATH, "a", encoding="utf-8") as fh:
                                        fh.write(json.dumps(metrics) + "\n")
                                except OSError as exc:
                                    logger.warning("Failed to write round metrics: %s", exc)

                                # Broadcast to /ws/metrics subscribers
                                await _broadcast(_metrics_subscribers, json.dumps(metrics))

                                logger.info(
                                    "Round %d metrics captured: mAP@50=%.4f",
                                    metrics["round"],
                                    metrics["mAP50"],
                                )
            except OSError as exc:
                logger.warning("Error reading server.log during monitoring: %s", exc)

            if retcode is not None:
                # Process has exited. Let's wait a little bit and read any final data, then break.
                await asyncio.sleep(0.5)
                try:
                    with open(server_log_path, "r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(offset)
                        new_data = fh.read()
                    if new_data:
                        for line in new_data.splitlines():
                            line = line.rstrip("\n\r")
                            if line:
                                await _broadcast(_logs_subscribers, line)
                                metrics = _parse_round_line(line)
                                if metrics:
                                    async with _state.lock:
                                        _state.current_round = metrics["round"]
                                    # Update Prometheus Metrics
                                    PROM_CURRENT_ROUND.set(metrics.get("round", 0))
                                    PROM_GLOBAL_MAP50.set(metrics.get("mAP50", 0.0))
                                    PROM_GLOBAL_LOSS.set(metrics.get("train_loss", metrics.get("eval_loss", 0.0)))
                                    PROM_GINI_COEFFICIENT.set(metrics.get("gini", 0.0))
                                    PROM_COMMUNICATION_MB.set(metrics.get("comm_MB", 0.0))
                                    PROM_ACTIVE_CLIENTS.set(metrics.get("num_clients", 1))
                                    try:
                                        with open(ROUND_METRICS_PATH, "a", encoding="utf-8") as fh:
                                            fh.write(json.dumps(metrics) + "\n")
                                    except OSError:
                                        pass
                                    await _broadcast(_metrics_subscribers, json.dumps(metrics))
                except OSError:
                    pass
                break

            await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        logger.info("Server output monitor cancelled.")
    except Exception as exc:
        logger.exception("Server output monitor error: %s", exc)
        async with _state.lock:
            _state.last_error = str(exc)
    finally:
        retcode = proc.poll()
        async with _state.lock:
            if _state.phase == TrainingPhase.RUNNING:
                PROM_CONNECTED_CLIENTS.set(0)
                if retcode is not None and retcode == 0:
                    _state.phase = TrainingPhase.COMPLETED
                    logger.info("Training completed (server exited 0).")
                elif retcode is not None:
                    _state.phase = TrainingPhase.FAILED
                    _state.last_error = f"Server exited with code {retcode}"
                    logger.error("Server exited with code %d.", retcode)


# ---------------------------------------------------------------------------
# Background: tail a file and broadcast new lines (for /ws/logs fallback)
# ---------------------------------------------------------------------------

async def _tail_file(
    path: pathlib.Path,
    subscribers: List[asyncio.Queue],
    poll_interval: float = 0.5,
) -> None:
    """Poll *path* for new lines and push them to *subscribers*.

    Used as a fallback log streamer when we need to watch existing log files
    rather than (or in addition to) server stdout.
    """
    offset = 0
    if path.is_file():
        try:
            offset = path.stat().st_size
        except OSError:
            pass

    try:
        while True:
            await asyncio.sleep(poll_interval)
            if not path.is_file():
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offset)
                    new_data = fh.read()
                    if new_data:
                        offset = fh.tell()
                        for line in new_data.splitlines():
                            if line.strip():
                                await _broadcast(subscribers, line)
            except OSError:
                continue
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _kill_process(proc: subprocess.Popen) -> None:
    """Terminate a subprocess, escalating to kill after a short wait."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    except OSError as exc:
        logger.warning("Error killing PID %s: %s", proc.pid, exc)


def _kill_all_training() -> None:
    """Kill server + all client subprocesses."""
    global _state
    if _state.server_process:
        _kill_process(_state.server_process)
    for cp in _state.client_processes:
        _kill_process(cp)


# ---------------------------------------------------------------------------
# Lifecycle events (Lifespan)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure required directories exist on server boot and clean up resources on shutdown."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("FedFOD API server started.")
    yield
    # Cleanup: kill training processes and cancel background tasks.
    _kill_all_training()
    if _state.monitor_task and not _state.monitor_task.done():
        _state.monitor_task.cancel()
    logger.info("FedFOD API server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FedFOD API Server",
    description="REST + WebSocket bridge for the FedFOD federated learning system.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())


# ===================================================================== #
#                       REST ENDPOINTS                                    #
# ===================================================================== #


# -- Status ------------------------------------------------------------- #

@app.get("/api/status", response_model=StatusResponse, tags=["Status"])
async def get_status():
    """Return current server status, uptime, and training info."""
    async with _state.lock:
        return StatusResponse(
            status="ok",
            uptime=round(time.time() - _server_start_time, 2),
            training_active=_state.phase == TrainingPhase.RUNNING,
            current_round=_state.current_round,
            connected_clients=len(_state.client_pids),
        )


# -- Global Config ------------------------------------------------------ #

@app.get("/api/config/global", tags=["Config"])
async def get_global_config():
    """Return the parsed contents of ``global_config.yaml``."""
    try:
        return _load_yaml(GLOBAL_CONFIG_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="global_config.yaml not found")


@app.put("/api/config/global", tags=["Config"])
async def update_global_config(body: Dict[str, Any]):
    """Overwrite ``global_config.yaml`` with the provided JSON body.

    The body should be a full or partial config dict. It will **replace**
    the entire YAML file.
    """
    async with _state.lock:
        if _state.phase == TrainingPhase.RUNNING:
            raise HTTPException(
                status_code=409,
                detail="Cannot update config while training is running.",
            )
    try:
        _save_yaml(GLOBAL_CONFIG_PATH, body)
        return {"status": "ok", "message": "Global config updated."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -- Airport Configs ---------------------------------------------------- #

@app.get("/api/config/airports", tags=["Config"])
async def list_airport_configs():
    """List all airport configuration files with their parsed contents."""
    if not AIRPORT_CONFIGS_DIR.is_dir():
        return {"airports": []}
    results = []
    for yaml_file in sorted(AIRPORT_CONFIGS_DIR.glob("*.yaml")):
        try:
            data = _load_yaml(yaml_file)
            results.append({
                "id": yaml_file.stem,
                **data
            })
        except Exception as exc:
            logger.warning("Error loading airport config %s: %s", yaml_file, exc)
    return {"airports": results}


@app.get("/api/config/airports/{airport_id}", tags=["Config"])
async def get_airport_config(airport_id: str):
    """Return a single airport config by ID (stem of the YAML filename).

    Example: ``/api/config/airports/airport_A``
    """
    path = AIRPORT_CONFIGS_DIR / f"{airport_id}.yaml"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Airport config '{airport_id}' not found.")
    try:
        return _load_yaml(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/api/config/airports/{airport_id}", tags=["Config"])
async def update_airport_config(airport_id: str, body: Dict[str, Any]):
    """Overwrite the airport config YAML with the provided JSON body."""
    async with _state.lock:
        if _state.phase == TrainingPhase.RUNNING:
            raise HTTPException(
                status_code=409,
                detail="Cannot update config while training is running.",
            )
    path = AIRPORT_CONFIGS_DIR / f"{airport_id}.yaml"
    try:
        _save_yaml(path, body)
        return {"status": "ok", "message": f"Airport config '{airport_id}' updated."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -- Checkpoints -------------------------------------------------------- #

@app.get("/api/checkpoints", tags=["Prediction"])
async def list_checkpoints():
    """List all available model checkpoints in the checkpoints directory."""
    checkpoints = []
    checkpoint_dir = BASE_DIR / "checkpoints"
    if checkpoint_dir.is_dir():
        for f in checkpoint_dir.glob("*.pt"):
            checkpoints.append(f"checkpoints/{f.name}")
    return sorted(checkpoints, reverse=True)


# -- Metrics ------------------------------------------------------------ #

@app.get("/api/metrics/report", tags=["Metrics"])
async def get_simulation_report():
    """Return the contents of ``logs/simulation_report.json``."""
    if not SIMULATION_REPORT_PATH.is_file():
        raise HTTPException(status_code=404, detail="simulation_report.json not found.")
    try:
        with open(SIMULATION_REPORT_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Error reading report: {exc}")


@app.get("/api/metrics/rounds", tags=["Metrics"])
async def get_round_metrics():
    """Return all per-round metrics from ``logs/round_metrics.jsonl``.

    Each line of the JSONL file is a JSON object with keys:
    ``round``, ``mAP50``, ``FAR_per_hr``, ``comm_MB``, ``time_s``.
    """
    if not ROUND_METRICS_PATH.is_file():
        return {"rounds": []}
    results: List[Dict[str, Any]] = []
    try:
        with open(ROUND_METRICS_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Error reading round metrics: {exc}")
    return {"rounds": results}


# -- Logs --------------------------------------------------------------- #

@app.get("/api/logs", tags=["Logs"])
async def get_logs(n: int = Query(default=200, ge=1, le=5000, description="Number of lines")):
    """Return the last *n* log lines from all log files in ``logs/``."""
    return {"lines": _collect_log_lines(n)}


# -- Training Control --------------------------------------------------- #

@app.post("/api/training/start", tags=["Training"])
async def start_training(req: TrainingStartRequest):
    """Start a federated learning training session.

    **Workflow**:
    1. Launch ``server_runner.py`` as a subprocess.
    2. Wait ~3 seconds for the Flower server to bind.
    3. Launch N ``client_runner.py`` subprocesses.
    4. Start a background task monitoring server stdout for round metrics.

    Returns immediately with process PIDs and training state.
    """
    global _state

    async with _state.lock:
        if _state.phase in (TrainingPhase.RUNNING, TrainingPhase.STARTING):
            raise HTTPException(
                status_code=409,
                detail=f"Training already {_state.phase.value}. Stop it first.",
            )
        _state.reset()
        _state.phase = TrainingPhase.STARTING
        _state.total_rounds = req.rounds
        _state.start_time = time.time()

    # -- Resolve clients --
    clients = req.clients
    if not clients:
        # Default: 3 clients using the three airport configs
        airport_files = sorted(AIRPORT_CONFIGS_DIR.glob("*.yaml"))
        clients = [
            ClientSpec(
                id=str(i),
                airport_config=str(af.relative_to(BASE_DIR)),
            )
            for i, af in enumerate(airport_files)
        ]
        if not clients:
            # Absolute fallback
            clients = [
                ClientSpec(id="0", airport_config="config/airport_configs/airport_A.yaml"),
                ClientSpec(id="1", airport_config="config/airport_configs/airport_B.yaml"),
                ClientSpec(id="2", airport_config="config/airport_configs/airport_N.yaml"),
            ]

    # -- Clear old round_metrics.jsonl --
    try:
        ROUND_METRICS_PATH.unlink(missing_ok=True)
    except OSError:
        pass

    # -- Build server command --
    grpc_port = req.port
    if grpc_port == 8000 or grpc_port == 8085:
        grpc_port = 8080

    # Ensure the chosen port is actually free, otherwise find the next free one
    import socket
    port_free = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", grpc_port))
            port_free = True
        except OSError:
            pass

    if not port_free:
        original_port = grpc_port
        for p in range(8080, 8180):
            if p == 8000 or p == 8085:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    grpc_port = p
                    break
                except OSError:
                    continue
        logger.warning(f"Port {original_port} was in use. Dynamically switched to free port {grpc_port}.")

    server_cmd = [
        sys.executable, str(SERVER_RUNNER),
        "--config", str(GLOBAL_CONFIG_PATH),
        "--port", str(grpc_port),
        "--rounds", str(req.rounds),
        "--min-clients", str(req.min_clients),
        "--backbone", req.backbone,
    ]
    if req.dummy_model:
        server_cmd.append("--dummy-model")

    logger.info("Starting server: %s", " ".join(server_cmd))

    # Force subprocesses to output UTF-8 encoding
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"

    try:
        server_log = open(LOGS_DIR / "server.log", "w", encoding="utf-8")
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            env=child_env,
        )
        server_log.close()
    except Exception as exc:
        async with _state.lock:
            _state.phase = TrainingPhase.FAILED
            _state.last_error = f"Failed to start server: {exc}"
        raise HTTPException(status_code=500, detail=str(exc))

    async with _state.lock:
        _state.server_process = server_proc

    # -- Wait for server to bind --
    await asyncio.sleep(8)

    # Check server is still alive
    if server_proc.poll() is not None:
        async with _state.lock:
            _state.phase = TrainingPhase.FAILED
            _state.last_error = "Server process exited before clients could connect."
        raise HTTPException(
            status_code=500,
            detail="server_runner.py exited prematurely.",
        )

    # -- Launch client processes --
    client_procs: List[subprocess.Popen] = []
    server_address = f"localhost:{grpc_port}"

    for spec in clients:
        client_cmd = [
            sys.executable, str(CLIENT_RUNNER),
            "--client-id", spec.id,
            "--server", server_address,
            "--config", str(GLOBAL_CONFIG_PATH),
            "--airport-config", str(BASE_DIR / spec.airport_config),
            "--backbone", req.backbone,
        ]
        if req.dummy_model:
            client_cmd.append("--dummy-model")

        logger.info("Starting client %s: %s", spec.id, " ".join(client_cmd))

        try:
            log_file = open(LOGS_DIR / f"client_{spec.id}.log", "w", encoding="utf-8")
            cp = subprocess.Popen(
                client_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
                env=child_env,
            )
            log_file.close()
            client_procs.append(cp)
        except Exception as exc:
            logger.error("Failed to start client %s: %s", spec.id, exc)

    async with _state.lock:
        _state.client_processes = client_procs
        _state.connected_clients = len(client_procs)
        PROM_CONNECTED_CLIENTS.set(len(client_procs))
        _state.phase = TrainingPhase.RUNNING

    # -- Start background monitor --
    monitor = asyncio.create_task(_monitor_server_output(server_proc))
    async with _state.lock:
        _state.monitor_task = monitor

    logger.info(
        "Training started: server PID=%d, %d clients, %d rounds.",
        server_proc.pid,
        len(client_procs),
        req.rounds,
    )

    return {
        "status": "started",
        "server_pid": server_proc.pid,
        "client_pids": [p.pid for p in client_procs],
        "rounds": req.rounds,
        "port": req.port,
    }


@app.post("/api/training/stop", tags=["Training"])
async def stop_training():
    """Stop all running training processes (server + clients)."""
    global _state

    async with _state.lock:
        if _state.phase not in (TrainingPhase.RUNNING, TrainingPhase.STARTING):
            raise HTTPException(
                status_code=409,
                detail=f"No active training to stop (phase={_state.phase.value}).",
            )
        _state.phase = TrainingPhase.STOPPING

    # Cancel the monitor task first
    if _state.monitor_task and not _state.monitor_task.done():
        _state.monitor_task.cancel()
        try:
            await _state.monitor_task
        except asyncio.CancelledError:
            pass

    # Kill processes
    _kill_all_training()

    async with _state.lock:
        stopped_round = _state.current_round
        _state.phase = TrainingPhase.IDLE
        _state.server_process = None
        _state.client_processes = []
        _state.monitor_task = None
        _state.connected_clients = 0
        PROM_CONNECTED_CLIENTS.set(0)

    logger.info("Training stopped at round %d.", stopped_round)

    return {
        "status": "stopped",
        "stopped_at_round": stopped_round,
    }


@app.get("/api/training/state", response_model=TrainingStateResponse, tags=["Training"])
async def get_training_state():
    """Return the current training lifecycle state."""
    async with _state.lock:
        # Refresh connected_clients count from live PIDs
        _state.connected_clients = len(_state.client_pids)
        PROM_CONNECTED_CLIENTS.set(_state.connected_clients)
        return _state.to_response()


# ===================================================================== #
#                      WEBSOCKET ENDPOINTS                                #
# ===================================================================== #


@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    """Stream round metrics in real-time.

    On connect, sends all existing metrics from ``round_metrics.jsonl``,
    then tails the file and pushes new lines as they appear.
    """
    await websocket.accept()
    logger.info("WebSocket /ws/metrics client connected.")

    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    _metrics_subscribers.append(queue)

    try:
        # Send existing metrics as backfill
        if ROUND_METRICS_PATH.is_file():
            try:
                with open(ROUND_METRICS_PATH, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            await websocket.send_text(line)
            except OSError:
                pass

        # Stream new metrics
        while True:
            message = await queue.get()
            await websocket.send_text(message)

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/metrics client disconnected.")
    except Exception as exc:
        logger.warning("WebSocket /ws/metrics error: %s", exc)
    finally:
        try:
            _metrics_subscribers.remove(queue)
        except ValueError:
            pass


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    """Stream log lines in real-time.

    On connect, sends the last 50 log lines as backfill, then pushes
    new lines from server stdout and log file tailing.
    """
    await websocket.accept()
    logger.info("WebSocket /ws/logs client connected.")

    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _logs_subscribers.append(queue)

    # Also start a background tail of simulation.log for non-training logs
    tail_task: Optional[asyncio.Task] = None
    sim_log = LOGS_DIR / "simulation.log"
    if sim_log.is_file():
        tail_task = asyncio.create_task(
            _tail_file(sim_log, [queue], poll_interval=1.0)
        )

    try:
        # Backfill last 50 lines
        for line in _collect_log_lines(50):
            await websocket.send_text(line)

        # Stream new lines
        while True:
            message = await queue.get()
            await websocket.send_text(message)

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/logs client disconnected.")
    except Exception as exc:
        logger.warning("WebSocket /ws/logs error: %s", exc)
    finally:
        if tail_task and not tail_task.done():
            tail_task.cancel()
        try:
            _logs_subscribers.remove(queue)
        except ValueError:
            pass


# ===================================================================== #
#                       WEB APP & DETECTOR APIs                          #
# ===================================================================== #

@app.post("/api/predict", tags=["Prediction"])
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("yolov8n.pt"),
    conf: float = Form(0.25),
    no_filter: bool = Form(True),
    rain_prob: float = Form(0.0),
    fog_prob: float = Form(0.0),
    glare_prob: float = Form(0.05),
    hour: float = Form(14.0),
    dynamic_mode: bool = Form(False),
    auto_weather: bool = Form(True)
):
    """Run runway FOD detection on an uploaded image.
    
    Draws highlighted bounding boxes and category labels on the image,
    and returns a base64-encoded version of the annotated image.
    """
    global _cached_detectors, _cached_clip_detector
    
    # Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")
        
    active_model = model_name
    h, w, _ = frame.shape
    
    # 1. Dynamic Mode - Choose best algorithm based on image resolution
    if dynamic_mode:
        if max(h, w) > 1000:
            checkpoint_path = BASE_DIR / "checkpoints" / "round_90.pt"
            active_model = "checkpoints/round_90.pt" if checkpoint_path.is_file() else "rtdetr-l.pt"
        else:
            active_model = "yolov8n.pt"

    # Get or lazily load the detector backbone
    cache_key = f"{active_model}_{conf}"
    if cache_key not in _cached_detectors:
        try:
            from src.client.inference import RTDETRDetector
            _cached_detectors[cache_key] = RTDETRDetector(
                model_path=active_model,
                conf_threshold=conf,
                device="cpu"  # Keep CPU-friendly to avoid thread blocking
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load detector backbone: {exc}")
            
    detector = _cached_detectors[cache_key]
    
    # 2. Auto Weather Detection
    detected_weather = {}
    if auto_weather:
        detected_weather = _analyze_image_weather(frame)
        active_rain = detected_weather["rain_prob"]
        active_fog = detected_weather["fog_prob"]
        active_glare = detected_weather["glare_prob"]
        active_hour = detected_weather["hour"]
        active_mean = detected_weather["luminance_mean"]
        active_std = detected_weather["luminance_std"]
    else:
        active_rain = rain_prob
        active_fog = fog_prob
        active_glare = glare_prob
        active_hour = hour
        active_mean = 0.45 if fog_prob > 0.5 else 0.65
        active_std = 0.05 if fog_prob > 0.5 else 0.12

    # 3. Core Object Detection
    detections = detector.detect(frame)
    
    # 4. Optional Weather-aware False Alarm Filter
    filtered_detections = detections
    if not no_filter:
        try:
            from src.client.inference import FalseAlarmFilterMLP
            weather_ctx = {
                "rain_prob": active_rain,
                "fog_prob": active_fog,
                "glare_prob": active_glare,
                "hour": active_hour,
                "luminance_mean": active_mean,
                "luminance_std": active_std
            }
            mlp = FalseAlarmFilterMLP(input_dim=12)
            mlp_path = BASE_DIR / "checkpoints" / "mlp_filter.pt"
            if mlp_path.is_file():
                mlp.load_model(str(mlp_path))
                logger.info("Loaded weather-aware False Alarm Filter weights from %s", mlp_path)
            else:
                logger.warning("No pre-trained False Alarm Filter weights found at %s; using initialized weights.", mlp_path)
            filtered_detections = mlp.filter_detections(detections, weather_ctx, frame)
        except Exception as exc:
            logger.warning("False alarm filter failed: %s", exc)
            
    # 3. Optional CLIP Verification
    results = []
    if filtered_detections:
        if _cached_clip_detector is None:
            try:
                from src.client.open_world import CLIPOpenWorldDetector
                _cached_clip_detector = CLIPOpenWorldDetector(device="cpu")
            except Exception as exc:
                logger.warning("CLIP detector failed to load: %s", exc)
        
        for idx, det in enumerate(filtered_detections):
            x1, y1, x2, y2 = map(int, det.bbox)
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            
            classification_str = det.class_name
            if crop.size > 0 and _cached_clip_detector is not None:
                try:
                    emb = _cached_clip_detector.compute_fod_embedding(crop)
                    _, class_name, sim = _cached_clip_detector.classify_known(emb)
                    classification_str = f"{class_name} ({sim:.2f})"
                except Exception:
                    pass
            
            results.append({
                "id": idx + 1,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(det.confidence, 4),
                "class_name": det.class_name,
                "clip_label": classification_str
            })
            
            # Draw bbox and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 64, 52), 2)  # Bright neon-red
            label = f"#{idx+1} {det.class_name} ({det.confidence:.2f})"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 64, 52), 2)
            
    # Active learning: save the image and labels to training datasets for both Client 0 and Client 1
    if results:
        try:
            timestamp = int(time.time() * 1000)
            img_name = f"active_learning_{timestamp}.png"
            txt_name = f"active_learning_{timestamp}.txt"
            
            h, w, _ = frame.shape
            
            # Decode clean image (without bounding boxes drawn) to save to training set
            clean_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            for client_name in ["airport_A", "airport_B"]:
                img_dir = BASE_DIR / "data" / client_name / "train" / "images"
                lbl_dir = BASE_DIR / "data" / client_name / "train" / "labels"
                
                # Create directories if they do not exist
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                
                # Save clean image file
                cv2.imwrite(str(img_dir / img_name), clean_frame)
                
                # Generate yolo format labels
                label_lines = []
                for det in filtered_detections:
                    cx1, cy1, cx2, cy2 = det.bbox
                    bw = cx2 - cx1
                    bh = cy2 - cy1
                    xc = (cx1 + bw / 2) / w
                    yc = (cy1 + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    label_lines.append(f"{det.class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                    
                (lbl_dir / txt_name).write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                
            logger.info("Active learning sample saved successfully: %s", img_name)
        except Exception as exc:
            logger.warning("Active learning saver encountered an error: %s", exc)

    # Encode frame to base64 PNG
    _, buffer = cv2.imencode('.png', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "success": True,
        "detections": results,
        "image_base64": f"data:image/png;base64,{base64_image}",
        "active_model": active_model,
        "auto_weather_detected": detected_weather if auto_weather else None
    }


@app.post("/api/dataset/upload", tags=["Dataset"])
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset ZIP archive in YOLOv8 format.
    
    Extracts and sets it up automatically for Airport A (Client 0) and Airport B (Client 1).
    """
    global _state
    async with _state.lock:
        if _state.phase == TrainingPhase.RUNNING:
            raise HTTPException(
                status_code=409,
                detail="Cannot upload dataset while training is running.",
            )
            
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP archives are supported.")
        
    temp_zip_path = BASE_DIR / "temp_uploaded_dataset.zip"
    
    try:
        # Save file to temp path
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract for both Client 0 and Client 1
        for dest_name in ["airport_A", "airport_B"]:
            dest_dir = BASE_DIR / "data" / dest_name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
                
            # Roboflow compatibility: rename 'valid' folder to 'val'
            valid_folder = dest_dir / "valid"
            val_folder = dest_dir / "val"
            if valid_folder.exists() and not val_folder.exists():
                valid_folder.rename(val_folder)
                
        # Clean up temp ZIP
        if temp_zip_path.exists():
            temp_zip_path.unlink()
            
        return {"success": True, "message": "Dataset ZIP successfully extracted and configured for Airport A & B!"}
    except Exception as exc:
        if temp_zip_path.exists():
            temp_zip_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {exc}")


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_dashboard():
    """Serve the root dashboard index.html."""
    index_path = BASE_DIR / "web" / "index.html"
    if not index_path.is_file():
        return """
        <html>
            <head><title>FedFOD Loading</title></head>
            <body style="background:#0b0c10; color:#45f3ff; font-family:sans-serif; text-align:center; padding-top:20%;">
                <h1>FedFOD Web App Dashboard is loading...</h1>
            </body>
        </html>
        """
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

# Mount the static directory to serve javascript and css assets
web_dir = BASE_DIR / "web"
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")


# ===================================================================== #
#                           ENTRYPOINT                                    #
# ===================================================================== #

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
