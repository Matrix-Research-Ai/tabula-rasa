# Tabula Rasa Dashboard — Architecture & Structure Analysis

**Version:** 1.3 | **Last Updated:** June 2026 | **Scope:** Frontend Dashboard & API Integration

---

## 1. Directory Structure

The dashboard is designed to be 100% offline, zero-dependency, and modular.
Legacy/orphaned files are preserved in `experimental/` rather than deleted.

```
C:\Users\Admin\tabula-rasa\Dashboard\

  core/
    dashboard.html          (8.3 KB)   MAIN ENTRY POINT: Shell, sidebar, iframe workspace

  views/
    interactive_chat.html        (10.5 KB)  INTERACTIVE CHAT: Bubbles, confidence, routing, auto-teach
    training_monitor.html (14.5 KB)  TRAINING MONITOR: Canvas chart, Piaget phases, live debug log
    memory_viewer.html      (8.7 KB)   MEMORY VIEWER: Session memory, corrections, search, raw JSON
    model_config.html       (9.0 KB)   V1.3 CONFIG: 25+ hyperparameters, architecture toggles
    specialist_trainer.html (10.2 KB)  SPECIALIST TRAINER: Launch per-specialist training, config
    checkpoint_manager.html (7.4 KB)   CHECKPOINT MANAGER: Browse .pt files, load, compare
    sleep_cycle.html        (11.6 KB)  SLEEP CYCLE: Egefalos consolidation pipeline visualization
    server_control.html     (7.1 KB)   SERVER CONTROL: Start/stop/restart services
    log_viewer.html         (3.6 KB)   LOG VIEWER: training.log tail viewer
    gpu_training.html       (15.8 KB)  GPU TRAINING: Build deployment package, upload to cloud
    system_status.html      (7.5 KB)   SYSTEM HEALTH: Egefalos brain, specialists table, resources

  (all files under views/, no experimental/ folder)

  serve.py                  (2.1 KB)   Standalone HTTP server for Dashboard/ (duplicate of root)
  STRUCTURE.md                        This file
```

### Path reference

| File | URL |
|------|-----|
| Shell sidebar | `http://localhost:8000/core/dashboard.html` |
| Interactive Chat | `../views/interactive_chat.html` |
| Training Monitor | `../views/training_monitor.html` |
| Memory Viewer | `../views/memory_viewer.html` |
| Model Config | `../views/model_config.html` |
| Specialist Trainer | `../views/specialist_trainer.html` |
| Checkpoint Manager | `../views/checkpoint_manager.html` |
| Sleep Cycle | `../views/sleep_cycle.html` |
| System Status | `../views/system_status.html` |
| Server Control | `../views/server_control.html` |
| GPU Training | `../views/gpu_training.html` |
| Log Viewer | `../views/log_viewer.html` |
| Whitepaper | `../views/whitepaper.html` |

> **Note:** All paths in `core/dashboard.html` use relative paths starting with
> `../views/` or `../experimental/` because the shell lives one directory deeper.

---

## 2. System Architecture

The dashboard operates as a decoupled frontend that communicates with two local
backend services via REST API.

```
                         ┌──────────────────────────────────────┐
                         │          dashboard.html              │
                         │   (Sidebar + iframe workspace)       │
                         │   polls /health every 10s            │
                         └────────────┬─────────────┬───────────┘
                                      │             │
                    ┌─────────────────┘             └─────────────────┐
                    ▼                                                   ▼
    ┌──────────────────────────────┐              ┌──────────────────────────────┐
    │    Child Pages (in iframe)   │              │    Child Pages (in iframe)   │
    │                              │              │                              │
    │  interactive_chat.html            │              │  system_status.html          │
    │    POST /ask   (port 8002)   │              │    GET /skills (port 8002)   │
    │    POST /generate (8000)     │              │                              │
    │                              │              │  training_monitor.html     │
    │  model_config.html           │              │    GET /training-progress    │
    │    POST /correct  (8000)     │              │    GET /health               │
    └──────────┬───────────────────┘              └──────────┬───────────────────┘
               │                                            │
               ▼                                            ▼
    ┌──────────────────────┐                   ┌──────────────────────┐
    │  serve.py :8000      │                   │  tabula_rasa.py :8002│
    │  Math Transformer    │                   │  Skill Router        │
    │  Dashboard static    │                   │  Specialist Manager  │
    │  /health, /generate  │                   │  /ask, /skills       │
    │  /correct, /train    │                   │  /memory, /health    │
    └──────────────────────┘                   └──────────────────────┘
```

Both servers use **ThreadedHTTPServer** (multi-threaded) — no more hanging from
concurrent requests.

---

## 3. File-by-File Analysis

| File | Role & State | V1.3 Features Present | Action Required |
|------|-------------|----------------------|-----------------|
| **core/dashboard.html** | Main Shell. Modern, clean, offline-compatible. Polls `/health`. | Sidebar navigation (SVG icons, main + experimental sections), server status dot, correction counter, auto-highlights active nav link. | Wire child pages to update parent correction count via `postMessage`. |
| **views/interactive_chat.html** | Interactive Chat. Chat bubbles, confidence scores, router tags. | Auto-teach toggle, quick-prompt chips, fallback to port 8000. | Wire the `/correct` endpoint to actually trigger backend fine-tuning. |
| **views/training_monitor.html** | Training Monitor. Zero-dep Canvas chart, Piaget phase, debug log. | BPE vocab tracker, color-coded logs, auto-scroll. | Ensure `/training-progress` returns graceful JSON when idle (not just empty). |
| **views/model_config.html** | V1.3 Control Panel. 25+ parameters across 3 sections. | Architecture presets, training algorithm toggles, save/reload buttons. | **HIGH PRIORITY:** Wire `POST /api/config` to actually write to `config.py`. |
| **views/system_status.html** | System Health. Egefalos brain cards, specialists table. | Hippocampus buffer, Neocortex sleep status, Pythagorean review stats. | Ensure port 8002 `/skills` endpoint returns accurate, real-time accuracy metrics. |
| **experimental/server_control.html** | Server Manager. Terminal log, start/stop buttons. CSS unified with V1.3 theme. | None (Legacy UI) | **Decision:** Either wire to `POST /api/training/start`/`stop` or leave as reference. |
| **experimental/gpu_training.html** | Cloud Launcher. Provider cards, pricing, setup. CSS unified with V1.3 theme. | None (Legacy UI) | Preserved for future GPU support. Tabula Rasa is CPU-only by design. |
| **experimental/log_viewer.html** | Log Tailer. Auto-refreshes every 5s. CSS unified with V1.3 theme. | None (Legacy UI) | Functionality overlaps with `training_monitor.html` debug log. Keep as standalone viewer. |
| **serve.py** (in Dashboard/) | HTTP Server. Serves static files. | ThreadedHTTPServer (prevents hanging). | **Delete** this duplicate. Use the main `serve.py` at the project root. |

---

## 4. API Contract (Expected Endpoints)

To make the frontend fully functional, the backend must expose these endpoints:

| Method | Endpoint | Purpose | Expected Response |
|--------|----------|---------|-------------------|
| GET | `/health` | Check if server is alive | `{"status":"ok","corrections":12}` |
| POST | `/ask` | Send query to AI | `{"answer":"46","confidence":95,"skill":"math/add"}` |
| POST | `/correct` | Auto-teach fine-tuning | `{"status":"logged","loss":0.023}` |
| GET | `/training-progress` | Live chart data | `{"step":14250,"loss":1.67,"accuracy":14.2}` |
| GET | `/api/config` | Load current config | `{"layers":8,"activation":"swiglu","use_reversed":true}` |
| POST | `/api/config` | Save new config | `{"status":"saved"}` |
| GET | `/skills` | List specialists | `[{"name":"math/add","status":"trained","accuracy":92.4}]` |

---

## 5. Prioritized Improvement Roadmap

### Phase 1: Quick Wins (No Backend Changes)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 1 | **Delete `Dashboard/serve.py`** — stale duplicate | Cleans confusion | 1 min |
| 2 | **CSS unified** across all 3 experimental files (server_control, gpu_training, log_viewer) now match V1.3 theme | Consistent look | DONE |
| 3 | **Organized into folders** — core/, views/, experimental/ | Clean workspace | DONE |
| 4 | **Sidebar updated** with SVG icons, "Experimental / Legacy" section, auto-active-highlight on iframe load | Professional nav | DONE |
| 5 | **Parse `training.log` directly** in `log_viewer.html` via `fetch('/training.log')` instead of a missing API | Makes log viewer work | 10 min |

### Phase 2: Medium Effort (Backend Wiring Required)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 5 | **Wire `model_config.html`:** Implement `GET /api/config` and `POST /api/config` in `serve.py` | Config UI becomes real | 2-3 hours |
| 6 | **Wire `interactive_chat.html` corrections:** Connect auto-teach to `POST /correct` endpoint | Auto-teach actually trains | 1-2 hours |
| 7 | **Consolidate logs:** Move `log_viewer.html` functionality into `training_monitor.html` debug log, then delete | One less file to maintain | 1 hour |

### Phase 3: Structural & Architectural Improvements

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 8 | **Extract shared JS** into `Dashboard/dashboard.js` — API polling, formatting, DOM helpers, `postMessage` to parent | -30% code, fewer bugs | 3-4 hours |
| 9 | **iframe communication:** Have `dashboard.html` poll `/health` once and broadcast to children via `postMessage()` | Less network overhead | 2-3 hours |
| 10 | **Delete `gpu_training.html`** — contradicts Tabula Rasa philosophy ("CPU only, no cloud, deliberate constraint") | Philosophical consistency | 1 min |

---

## 6. Philosophical Alignment Notes

- **Tabula Rasa is CPU-only, no GPU, no cloud.** The `gpu_training.html` file directly
  contradicts the project's core philosophy and should be removed.
- **Zero external dependencies** is a deliberate design choice — the AI learns from
  nothing, and so does the UI. No CDNs, no build steps, no frameworks.
- **Local-first, offline-capable** — the dashboard must work with only the two Python
  servers running on localhost, no internet required.
