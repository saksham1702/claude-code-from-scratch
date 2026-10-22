#!/usr/bin/env python3
"""
s25_token_cost_ledger.py: Fine-Grained Token Accounting & Cost Attribution.

Motto: "Every token counted; every dollar attributed"

This module introduces a production cost ledger that accounts for token consumption
across conversation turns, prompt cache operations, and tool executions.

Key Architectural Concepts:
    1. Granular Token Categorization: Distinguishes between uncached input,
       cache write (creation), cache read (hit), and completion tokens.
    2. Model-Specific Pricing Rates: Dynamically calculates real USD spend based
       on Anthropic pricing tables (Sonnet, Haiku, Opus).
    3. Tool Cost Attribution: Measures context growth and cost contribution per tool.
    4. Audit Ledger and Reporting: Provides CLI commands (:cost, :tools, :export)
       and live turn cost telemetry.
"""

# Standard library imports
import asyncio
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union

# Local module imports
from core import (
    client,
    MODEL,
    EXTENDED_TOOLS,
    ASYNC_DISPATCH,
)


@dataclass
class ModelRate:
    """Stores token pricing per million tokens in USD."""
    input_rate: float
    cache_write_rate: float
    cache_read_rate: float
    output_rate: float


# Pricing table in USD per 1M tokens
PRICING_TABLE: Dict[str, ModelRate] = {
    "claude-3-5-sonnet": ModelRate(
        input_rate=3.00,
        cache_write_rate=3.75,
        cache_read_rate=0.30,
        output_rate=15.00
    ),
    "claude-3-7-sonnet": ModelRate(
        input_rate=3.00,
        cache_write_rate=3.75,
        cache_read_rate=0.30,
        output_rate=15.00
    ),
    "claude-3-5-haiku": ModelRate(
        input_rate=0.80,
        cache_write_rate=1.00,
        cache_read_rate=0.08,
        output_rate=4.00
    ),
    "claude-3-opus": ModelRate(
        input_rate=15.00,
        cache_write_rate=18.75,
        cache_read_rate=1.50,
        output_rate=75.00
    ),
}

DEFAULT_RATE: ModelRate = PRICING_TABLE["claude-3-5-sonnet"]


def resolve_model_rate(model_name: str) -> ModelRate:
    """
    Finds the matching pricing tier for a given model string.

    Args:
        model_name: Full model ID string.

    Returns:
        Matched ModelRate object.
    """
    for prefix, rate in PRICING_TABLE.items():
        if prefix in model_name.lower():
            return rate
    return DEFAULT_RATE


@dataclass
class TurnRecord:
    """Stores token metrics and cost calculations for a single turn."""
    turn_id: int
    timestamp: float
    model: str
    uncached_input: int
    cache_write: int
    cache_read: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    saved_usd: float
    tools_called: List[str] = field(default_factory=list)


class TokenCostLedger:
    """Maintains token consumption history, cost accounting, and tool attribution."""

    def __init__(self, model_name: str = MODEL) -> None:
        """Initializes the ledger container and counters."""
        self.model_name = model_name
        self.records: List[TurnRecord] = []
        self.tool_invocations: Dict[str, int] = {}
        self.tool_attributed_tokens: Dict[str, int] = {}

    def record_turn(
        self,
        turn_id: int,
        usage: Any,
        tools_called: List[str]
    ) -> TurnRecord:
        """
        Parses Anthropic API usage metadata and computes financial metrics.

        Args:
            turn_id: Sequence number of the interaction turn.
            usage: Anthropic response usage object.
            tools_called: Names of tools called in this turn.

        Returns:
            Calculated TurnRecord entry.
        """
        rate = resolve_model_rate(self.model_name)

        uncached_in = getattr(usage, "input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        out_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = uncached_in + cache_write + cache_read + out_tokens

        # Calculating actual billed cost
        cost_usd = (
            (uncached_in * rate.input_rate / 1_000_000.0)
            + (cache_write * rate.cache_write_rate / 1_000_000.0)
            + (cache_read * rate.cache_read_rate / 1_000_000.0)
            + (out_tokens * rate.output_rate / 1_000_000.0)
        )

        # Calculating savings compared to paying standard input rates on cached tokens
        baseline_cached_cost = cache_read * rate.input_rate / 1_000_000.0
        actual_cached_cost = cache_read * rate.cache_read_rate / 1_000_000.0
        saved_usd = max(0.0, baseline_cached_cost - actual_cached_cost)

        record = TurnRecord(
            turn_id=turn_id,
            timestamp=time.time(),
            model=self.model_name,
            uncached_input=uncached_in,
            cache_write=cache_write,
            cache_read=cache_read,
            output_tokens=out_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            saved_usd=saved_usd,
            tools_called=copy.copy(tools_called)
        )
        self.records.append(record)

        # Attributing usage to invoked tools
        if tools_called:
            tokens_per_tool = max(1, total_tokens // len(tools_called))
            for tool in tools_called:
                self.tool_invocations[tool] = self.tool_invocations.get(tool, 0) + 1
                self.tool_attributed_tokens[tool] = (
                    self.tool_attributed_tokens.get(tool, 0) + tokens_per_tool
                )

        return record

    def render_turn_ticker(self, record: TurnRecord) -> str:
        """
        Builds a compact one-line telemetry readout for the current turn.

        Args:
            record: Latest TurnRecord entry.

        Returns:
            Formatted ANSI telemetry string.
        """
        tools_str = f"tools: {','.join(record.tools_called)}" if record.tools_called else "no tools"
        return (
            f"\033[90m  [ledger] turn {record.turn_id} | "
            f"in: {record.uncached_input:,} | cache-w: {record.cache_write:,} | "
            f"cache-r: {record.cache_read:,} | out: {record.output_tokens:,} | "
            f"cost: ${record.cost_usd:.5f} | saved: ${record.saved_usd:.5f} | {tools_str}\033[0m"
        )

    def render_cost_table(self) -> str:
        """
        Generates an audit table summarizing turn-by-turn metrics and totals.

        Returns:
            Tabular markdown string.
        """
        if not self.records:
            return "Ledger is empty. No turns recorded yet."

        total_in = sum(r.uncached_input for r in self.records)
        total_cw = sum(r.cache_write for r in self.records)
        total_cr = sum(r.cache_read for r in self.records)
        total_out = sum(r.output_tokens for r in self.records)
        total_tokens = sum(r.total_tokens for r in self.records)
        total_cost = sum(r.cost_usd for r in self.records)
        total_saved = sum(r.saved_usd for r in self.records)

        lines = [
            "\n=== TOKEN ACCOUNTING & COST LEDGER ===",
            f"Model: {self.model_name}",
            f"{'Turn':<6}{'Input':<10}{'Cache-W':<10}{'Cache-R':<10}{'Output':<10}{'Cost (USD)':<14}{'Saved (USD)':<14}{'Tools'}",
            "-" * 80
        ]

        for r in self.records:
            tool_summary = ",".join(r.tools_called) if r.tools_called else "-"
            lines.append(
                f"{r.turn_id:<6}{r.uncached_input:<10}{r.cache_write:<10}{r.cache_read:<10}"
                f"{r.output_tokens:<10}${r.cost_usd:<13.5f}${r.saved_usd:<13.5f}{tool_summary}"
            )

        lines.append("-" * 80)
        lines.append(
            f"{'TOTAL':<6}{total_in:<10}{total_cw:<10}{total_cr:<10}"
            f"{total_out:<10}${total_cost:<13.5f}${total_saved:<13.5f}"
        )
        lines.append(f"Cumulative Tokens Processed: {total_tokens:,}")
        lines.append(f"Total Session Cost:          ${total_cost:.5f}")
        lines.append(f"Prompt Cache Savings:        ${total_saved:.5f}\n")

        return "\n".join(lines)

    def render_tool_attribution(self) -> str:
        """
        Builds a summary of tool invocation frequencies and token shares.

        Returns:
            Tabular tool attribution string.
        """
        if not self.tool_invocations:
            return "No tool calls recorded yet."

        lines = [
            "\n=== TOOL USAGE & COST ATTRIBUTION ===",
            f"{'Tool Name':<20}{'Calls':<10}{'Est. Tokens Share'}",
            "-" * 50
        ]

        for tool, count in sorted(self.tool_invocations.items(), key=lambda x: x[1], reverse=True):
            tokens = self.tool_attributed_tokens.get(tool, 0)
            lines.append(f"{tool:<20}{count:<10}{tokens:,}")

        lines.append("-" * 50)
        return "\n".join(lines)

    def export_json(self, destination_path: str) -> str:
        """
        Serializes the ledger records and attribution data into a JSON file.

        Args:
            destination_path: Filesystem path to export to.

        Returns:
            Confirmation message.
        """
        export_data = {
            "model": self.model_name,
            "export_timestamp": time.time(),
            "total_turns": len(self.records),
            "total_cost_usd": sum(r.cost_usd for r in self.records),
            "total_saved_usd": sum(r.saved_usd for r in self.records),
            "tool_invocations": self.tool_invocations,
            "tool_tokens": self.tool_attributed_tokens,
            "records": [asdict(r) for r in self.records]
        }

        os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
        with open(destination_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        return f"Ledger exported successfully to {destination_path}"


# Instantiating global ledger
ledger = TokenCostLedger()

# System instructions
SYSTEM: str = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use tools to solve tasks. Be efficient and avoid redundant tool calls. "
    "Every token spent is audited and tracked in the cost ledger."
)


async def dispatch_single_tool(block: Any) -> Tuple[str, str]:
    """
    Executes a single tool asynchronously and returns the output.

    Args:
        block: The tool_use content block.

    Returns:
        Tuple of tool_use_id and string output.
    """
    tool_name = block.name
    tool_input = block.input
    handler = ASYNC_DISPATCH.get(tool_name)

    if not handler:
        return block.id, f"Error: Unknown tool '{tool_name}'"

    first_val = str(list(tool_input.values())[0])[:80] if tool_input else ""
    print(f"\033[33m[{tool_name}] {first_val}...\033[0m")

    try:
        output = await handler(tool_input)
    except Exception as exc:
        output = f"Execution Error: {exc}"

    print(str(output)[:200])
    return block.id, str(output)


async def agent_loop_with_ledger(
    messages: List[Dict[str, Any]],
    turn_counter: int
) -> int:
    """
    Autonomous agent loop that monitors and accounts for token usage.

    Args:
        messages: Conversation message history.
        turn_counter: Current sequence number.

    Returns:
        Updated turn sequence number.
    """
    current_turn = turn_counter

    while True:
        current_turn += 1
        print("\n\033[36m> Thinking...\033[0m")

        def _blocking_stream_call():
            # Streaming response from API
            with client.messages.stream(
                model=MODEL,
                system=SYSTEM,
                messages=messages,
                tools=EXTENDED_TOOLS,
                max_tokens=8000,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                return stream.get_final_message()

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _blocking_stream_call)
        print()

        messages.append({"role": "assistant", "content": response.content})

        # Extracting tools called in this turn
        tool_blocks = [
            b for b in response.content
            if getattr(b, "type", None) == "tool_use"
        ]
        tool_names = [b.name for b in tool_blocks]

        # Accounting token usage into ledger
        if hasattr(response, "usage") and response.usage:
            record = ledger.record_turn(current_turn, response.usage, tool_names)
            print(ledger.render_turn_ticker(record))

        if response.stop_reason != "tool_use":
            return current_turn

        execution_pairs = await asyncio.gather(
            *[dispatch_single_tool(b) for b in tool_blocks]
        )
        results_map = dict(execution_pairs)
        turn_results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": results_map.get(b.id, "")}
            for b in tool_blocks
        ]
        messages.append({"role": "user", "content": turn_results})


async def main() -> None:
    """
    Main REPL loop supporting token accounting commands.
    """
    print("\033[90ms25: token cost ledger | per-turn accounting, cache ROI & tool attribution\033[0m")
    print("\033[90mCommands: :cost (audit table) | :tools (breakdown) | :export <file.json>\033[0m\n")

    history: List[Dict[str, Any]] = []
    turn_counter = 0
    loop = asyncio.get_event_loop()

    while True:
        try:
            query: str = await loop.run_in_executor(
                None, lambda: input("\033[36ms25 >> \033[0m").strip()
            )
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session.")
            break

        if not query or query.lower() in ("q", "exit", "quit"):
            break

        if query == ":cost":
            print(ledger.render_cost_table())
            continue

        if query == ":tools":
            print(ledger.render_tool_attribution())
            continue

        if query.startswith(":export"):
            parts = query.split(maxsplit=1)
            target_path = parts[1] if len(parts) > 1 else "cost_ledger_audit.json"
            print(ledger.export_json(target_path))
            continue

        history.append({"role": "user", "content": query})
        turn_counter = await agent_loop_with_ledger(history, turn_counter)
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
