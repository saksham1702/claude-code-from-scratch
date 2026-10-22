# Claude Code From Scratch

A deep-dive personal project reverse-engineering the harness engineering and system architecture behind modern AI coding agents like Claude Code. 

This repository documents my journey building a full production-grade coding harness from the ground up, starting from a minimal 20-line perception-action loop and scaling step-by-step into a 25-session multi-agent architecture.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange?style=for-the-badge)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## Why I Built This

Most discussions around AI coding assistants focus on prompt engineering or model benchmarks. However, the real engineering challenge that enables an assistant to reliably edit complex codebases across hours of conversation lies in the **harness** — the system software surrounding the model.

A robust harness must solve hard systems problems:
- Preventing context window degradation across long sessions
- Enforcing declarative permission and security boundaries
- Managing background tasks and parallel tool execution
- Isolating exploratory work into sandboxed contexts and Git worktrees
- Caching prompt prefixes and tracking token accounting down to the cent

Rather than relying on high-level multi-agent frameworks that obscure these primitives, I built this repository to implement and document each architectural layer from scratch in clean, readable Python.

---

## Core Engineering Principles

1. **No Framework Lock-in**: Everything is built with standard Python and direct API calls, eliminating black-box abstractions.
2. **Single Responsibility Per Stage**: `core.py` provides the reusable foundations (tools, dispatch tables, permissions), while each session file (`s01` through `s25`) introduces and demonstrates exactly one new architectural mechanism.
3. **The Model Decides, The Harness Executes**: The model is the sole engine for reasoning and decisions. The harness never second-guesses or overrides model logic; it executes tool calls, enforces safety policies, and structures the environment.
4. **Context is a Managed Asset**: Unbounded context leads to reasoning drift and excessive latency. History is deliberately compressed, subagents run in isolated environments, and cache prefixes are carefully structured.

---

## Repository Structure

```
claude-code-from-scratch/
│
├── core.py                          # Shared foundation: tools, dispatch map, permissions, loop
│
├── 01_perception_action_loop.py     # Stage 01: The minimal autonomous while loop
├── s02_tool_use.py                  # Stage 02: Tool dispatch map pattern
├── s03_todo_write.py                # Stage 03: Structured planning before execution
├── s04_subagent.py                  # Stage 04: Subagent context isolation
│
├── s05_skill_loading.py             # Stage 05: On-demand dynamic skill injection
├── s06_context_compact.py           # Stage 06: Three-layer context compression and memory
├── s07_task_system.py               # Stage 07: File-persisted task dependency graph
│
├── s08_background_tasks.py          # Stage 08: Non-blocking background task execution
├── s09_agent_teams.py               # Stage 09: Persistent agent teammates with JSONL mailboxes
├── s10_team_protocols.py            # Stage 10: Finite-state machine communication protocol
├── s11_autonomous_agents.py         # Stage 11: Autonomous task board self-assignment
├── s12_worktree_task_isolation.py   # Stage 12: Basic Git worktree task isolation
│
├── s13_streaming.py                 # Stage 13: Real-time token streaming interface
├── s14_tools_extended.py            # Stage 14: Extended tool arsenal and file rollback snapshots
├── s15_permissions.py               # Stage 15: YAML declarative rule-based permission system
├── s16_event_bus.py                 # Stage 16: Event bus and lifecycle hooks
├── s17_session_management.py        # Stage 17: Session persistence, branching, and resume
│
├── s18_parallel_tools.py            # Stage 18: Parallel tool execution with asyncio.gather
├── s19_interrupts.py                # Stage 19: Real-time interactive user interrupts
├── s20_cache_optimization.py        # Stage 20: Anthropic prompt caching prefix optimization
├── s21_mcp_runtime.py               # Stage 21: Model Context Protocol (MCP) server integration
│
├── s22_production_mailbox.py        # Stage 22: Redis pub/sub multi-agent mailboxes
├── s23_worktree_advanced.py         # Stage 23: Production Git worktree lifecycle management
├── s24_parallel_subagents.py        # Stage 24: Concurrent subagent exploration (fan-out fan-in)
├── s25_token_cost_ledger.py         # Stage 25: Fine-grained token accounting and cost tracking
│
├── config/
│   ├── permissions.yaml             # Declarative security rules (allow, ask, deny)
│   └── mcp_config.yaml              # External MCP server registry
│
├── skills/
│   ├── agent-builder/SKILL.md       # Agent architecture design patterns
│   ├── code-review/SKILL.md         # Structured code review methodology
│   └── pdf/SKILL.md                 # Document parsing and analysis guidance
│
├── litellm_config.yaml              # Optional proxy configuration for alternative models
└── requirements.txt                 # Project dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- An Anthropic API key (or alternative LLM API key via LiteLLM)
- Git installed on your system

### Installation

1. Clone the repository:
```bash
git clone https://github.com/saksham1702/claude-code-from-scratch.git
cd claude-code-from-scratch
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Open `.env` and configure your credentials:
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
MODEL_ID=claude-3-5-sonnet-20241022
```

---

## Running Alternative Models (Optional)

If you prefer to run local models or third-party providers (OpenAI, DeepSeek, Groq, Ollama) instead of Anthropic directly, you can use the built-in LiteLLM proxy:

1. Install LiteLLM proxy:
```bash
pip install "litellm[proxy]"
```

2. Configure your model in `litellm_config.yaml`:
```yaml
model_list:
  - model_name: local-model
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
```

3. Start the proxy server:
```bash
litellm --config litellm_config.yaml --port 4000
```

4. Point your `.env` to the local proxy:
```env
ANTHROPIC_BASE_URL=http://localhost:4000
ANTHROPIC_API_KEY=dummy-token-for-proxy
MODEL_ID=local-model
```

All 25 sessions run against the proxy without any code modifications.

---

## The 25-Stage Architectural Progression

The project is structured into six progressive phases, taking the harness from a basic script to an enterprise-grade agent runtime:

### Phase 1: The Core Perception-Action Loop (Stages 01–04)

Builds the foundational agent loop: sending messages, parsing structured tool requests, invoking handlers, and feeding outputs back to the model.

| Stage | Script | Core Mechanism |
|---|---|---|
| 01 | `01_perception_action_loop.py` | Minimal while loop driving model reasoning and tool dispatch |
| 02 | `s02_tool_use.py` | Extensible tool dispatch registry pattern |
| 03 | `s03_todo_write.py` | Upfront task planning and execution tracking |
| 04 | `s04_subagent.py` | Delegating heavy subtasks to an isolated child context |

```bash
python 01_perception_action_loop.py
python s02_tool_use.py
python s03_todo_write.py
python s04_subagent.py
```

---

### Phase 2: Knowledge & Context Management (Stages 05–07)

Prevents reasoning decay across long sessions through dynamic skill loading and context pruning.

| Stage | Script | Core Mechanism |
|---|---|---|
| 05 | `s05_skill_loading.py` | Dynamic on-demand loading of specialized SKILL.md documents |
| 06 | `s06_context_compact.py` | 3-layer context compression: micro-summaries, eviction, persistent disk memory |
| 07 | `s07_task_system.py` | File-backed task dependency graph supporting complex multi-step workflows |

```bash
python s05_skill_loading.py
python s06_context_compact.py
python s07_task_system.py
```

---

### Phase 3: Async Execution & Multi-Agent Collaboration (Stages 08–12)

Breaks out of synchronous execution with background daemon tasks, peer agent teams, and git worktrees.

| Stage | Script | Core Mechanism |
|---|---|---|
| 08 | `s08_background_tasks.py` | Asynchronous task execution with a thread-safe notification queue |
| 09 | `s09_agent_teams.py` | Multi-agent collaboration with persistent JSONL communication mailboxes |
| 10 | `s10_team_protocols.py` | Finite-state machine communication protocol ensuring coordination order |
| 11 | `s11_autonomous_agents.py` | Decentralized task self-assignment from a shared task board |
| 12 | `s12_worktree_task_isolation.py` | Sandboxed git worktrees allowing parallel branches without workspace collision |

```bash
python s08_background_tasks.py
python s09_agent_teams.py
python s10_team_protocols.py
python s11_autonomous_agents.py
python s12_worktree_task_isolation.py
```

---

### Phase 4: Production Hardening & Security (Stages 13–17)

Adds the guardrails and reliability features necessary for real developer workflows.

| Stage | Script | Core Mechanism |
|---|---|---|
| 13 | `s13_streaming.py` | Real-time token streaming for low-latency terminal interaction |
| 14 | `s14_tools_extended.py` | Production tool suite (read, write, grep, glob) with file snapshot rollback |
| 15 | `s15_permissions.py` | Declarative 3-tier security policy (`always_allow`, `ask_user`, `always_deny`) |
| 16 | `s16_event_bus.py` | Centralized lifecycle event bus with pre/post execution hooks |
| 17 | `s17_session_management.py` | Session persistence, branching (`:fork`), and resuming (`:resume`) |

```bash
python s13_streaming.py
python s14_tools_extended.py
python s15_permissions.py
python s16_event_bus.py
python s17_session_management.py
```

---

### Phase 5: High-Performance Async Runtime (Stages 18–21)

Optimizes throughput, latency, and tool extensibility.

| Stage | Script | Core Mechanism |
|---|---|---|
| 18 | `s18_parallel_tools.py` | Concurrent tool execution with `asyncio.gather` |
| 19 | `s19_interrupts.py` | Signal handling allowing mid-turn user steering and interruption |
| 20 | `s20_cache_optimization.py` | Anthropic KV prompt cache optimization with hit/miss tracking |
| 21 | `s21_mcp_runtime.py` | Model Context Protocol (MCP) integration with auto-tool registration |

```bash
python s18_parallel_tools.py
python s19_interrupts.py
python s20_cache_optimization.py
python s21_mcp_runtime.py
```

---

### Phase 6: Enterprise Isolation & Cost Governance (Stages 22–25)

Upgrades the system to production-ready multi-agent execution, concurrent exploration, and financial auditability.

| Stage | Script | Core Mechanism |
|---|---|---|
| 22 | `s22_production_mailbox.py` | Redis pub/sub messaging channels replacing local file mailboxes |
| 23 | `s23_worktree_advanced.py` | Full git worktree lifecycle: conflict detection, stale pruning, atomic cleanup |
| 24 | `s24_parallel_subagents.py` | Fan-out fan-in concurrent subagent exploration across multiple files |
| 25 | `s25_token_cost_ledger.py` | Granular token accounting, cache ROI analysis, and tool cost attribution |

```bash
python s22_production_mailbox.py
python s23_worktree_advanced.py
python s24_parallel_subagents.py
python s25_token_cost_ledger.py
```

---

## Key Technical Deep Dives

### Declarative Security Policy (`config/permissions.yaml`)

Tool calls go through declarative validation before hitting the operating system:
- **`always_deny`**: Automatically blocks dangerous commands (e.g. `rm -rf /`, `sudo`, piping downloads to bash).
- **`ask_user`**: Prompts the user with an interactive terminal confirmation for destructive actions (e.g. deletions, git commits, package installations).
- **`always_allow`**: Fast-paths non-destructive queries (e.g. `git status`, file reads, directory listings).

### Advanced Worktree Management (`s23_worktree_advanced.py`)

When agents execute code changes concurrently, modifying the main working directory causes conflicts. The advanced worktree system provides:
- Clean verification against dirty local trees and detached HEAD states
- Safe path sanitization to generate valid branch identifiers
- Automated pruning of orphaned metadata from terminated sessions
- Post-execution diff comparison to detect multi-agent merge conflicts

### Fan-Out Fan-In Parallel Subagents (`s24_parallel_subagents.py`)

Rather than investigating files sequentially, the Lead Agent dispatches multiple isolated subagents concurrently via `asyncio.gather`:
- Each subagent runs in its own private context with dedicated tool handlers.
- Exploratory logs and failed read attempts do not pollute the parent context window.
- The subagents synthesize their conclusions into a combined report for the Lead Agent.

### Fine-Grained Token Cost Ledger (`s25_token_cost_ledger.py`)

Audits token spend and measures the financial benefit of prompt caching:
- Distinguishes base input tokens, cache writes, cache read hits, and completion tokens.
- Calculates exact dollar costs across model tiers (Claude 3.5 Sonnet, Haiku, Opus).
- Provides interactive commands (`:cost`, `:tools`, `:export`) for session auditing.

---

## Future Improvements

Features I plan to experiment with in subsequent milestones:
1. **ChromaDB Vector Store**: Replacing flat markdown memory files with vector embeddings for semantic retrieval across long historical sessions.
2. **Webhook Event Sinks**: Forwarding lifecycle events to external endpoints (Slack, Discord, webhooks) for remote notifications.
3. **Automated Benchmark Harness**: Building an LLM-as-a-judge evaluation suite to measure agent task completion accuracy and tool call efficiency.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
