#!/usr/bin/env python3
"""
s24_parallel_subagents.py: Fan-Out Fan-In Parallel Subagent Exploration.

Motto: "Explore multiple paths at once; isolate contexts in parallel"

This module implements parallel subagent spawning using asyncio.gather.
When facing large exploration tasks, multi-file code reviews, or simultaneous
investigations, the Lead Agent dispatches multiple subagents concurrently.

Key Architectural Concepts:
    1. Parallel Fan-Out: The lead agent dispatches N subagents in a single tool call.
    2. Context Isolation: Each subagent operates inside a private message history.
    3. Non-Blocking Concurrency: Subagent loops execute concurrently with asyncio.gather.
    4. Fan-In Aggregation: Results from all subagents are gathered and summarized
       into a unified report returned to the Lead Agent.
"""

# Standard library imports
import asyncio
import copy
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

# Local module imports
from core import (
    client,
    MODEL,
    EXTENDED_TOOLS,
    ASYNC_DISPATCH,
)

# Lead Agent system instructions
SYSTEM: str = (
    f"You are a lead coding agent at {os.getcwd()}. "
    "When investigating multiple files, exploring different hypotheses, or dividing complex work, "
    "use spawn_parallel_subagents to explore all paths concurrently. "
    "Synthesize the returned subagent findings into a clear, cohesive final response."
)

# Specialist Subagent system instructions
SUBAGENT_SYSTEM: str = (
    f"You are an isolated exploration subagent at {os.getcwd()}. "
    "Investigate your assigned subtask thoroughly using available tools. "
    "Provide a concise, factual summary of findings and conclusions at the end."
)


@dataclass
class SubtaskResult:
    """Stores the execution outcome of an isolated subagent."""
    task_id: str
    prompt: str
    summary: str
    tool_calls_count: int = 0
    is_success: bool = True
    error_message: Optional[str] = None


async def execute_subagent_tool(block: Any) -> Tuple[str, str]:
    """
    Executes a single tool requested by a subagent.

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

    try:
        output = await handler(tool_input)
        return block.id, str(output)
    except Exception as exc:
        return block.id, f"Tool execution failed: {exc}"


def build_tool_results_list(
    tool_blocks: List[Any],
    execution_pairs: List[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """
    Constructs tool_result blocks from execution outputs.

    Args:
        tool_blocks: List of requested tool blocks.
        execution_pairs: List of (tool_use_id, output) tuples.

    Returns:
        Formatted list of tool_result message dictionaries.
    """
    results_map = dict(execution_pairs)
    return [
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": results_map.get(block.id, ""),
        }
        for block in tool_blocks
    ]


def extract_subagent_summary(messages: List[Dict[str, Any]]) -> str:
    """
    Extracts the final text summary from the subagent conversation history.

    Args:
        messages: Conversation history of the subagent.

    Returns:
        Concatenated text response.
    """
    if not messages:
        return "(no output produced)"

    last_content = messages[-1].get("content", [])
    if isinstance(last_content, str):
        return last_content

    text_parts = [
        getattr(block, "text", "")
        for block in last_content
        if getattr(block, "type", None) == "text" or hasattr(block, "text")
    ]
    return "".join(text_parts).strip() or "(subtask finished without text summary)"


async def run_single_subagent(task_id: str, prompt: str) -> SubtaskResult:
    """
    Executes a single subagent loop inside an isolated context.

    Args:
        task_id: Unique identifier for the subtask.
        prompt: Task instructions for the subagent.

    Returns:
        SubtaskResult containing the summary and stats.
    """
    print(f"\033[35m  [subagent:{task_id}] started: {prompt[:60]}...\033[0m")
    sub_messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_counter = 0

    while True:
        def _invoke_api():
            # Calling messages API synchronously in background thread
            return client.messages.create(
                model=MODEL,
                system=SUBAGENT_SYSTEM,
                messages=sub_messages,
                tools=EXTENDED_TOOLS,
                max_tokens=4000,
            )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, _invoke_api)
        except Exception as exc:
            print(f"\033[31m  [subagent:{task_id}] error: {exc}\033[0m")
            return SubtaskResult(
                task_id=task_id,
                prompt=prompt,
                summary="",
                tool_calls_count=tool_counter,
                is_success=False,
                error_message=str(exc)
            )

        sub_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_blocks = [
            block for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]
        tool_counter += len(tool_blocks)

        execution_pairs = await asyncio.gather(
            *[execute_subagent_tool(block) for block in tool_blocks]
        )
        tool_results = build_tool_results_list(tool_blocks, execution_pairs)
        sub_messages.append({"role": "user", "content": tool_results})

    summary = extract_subagent_summary(sub_messages)
    print(f"\033[35m  [subagent:{task_id}] completed ({tool_counter} tools used)\033[0m")

    return SubtaskResult(
        task_id=task_id,
        prompt=prompt,
        summary=summary,
        tool_calls_count=tool_counter,
        is_success=True
    )


def format_subagent_report(results: List[SubtaskResult]) -> str:
    """
    Formats the aggregated findings into a structured markdown report.

    Args:
        results: List of completed SubtaskResult objects.

    Returns:
        Markdown string aggregating all subagent findings.
    """
    report_lines = [
        f"### Parallel Subagents Report ({len(results)} tasks executed concurrently)\n"
    ]

    for item in results:
        status_tag = "SUCCESS" if item.is_success else f"FAILED: {item.error_message}"
        report_lines.append(f"#### Subtask: {item.task_id} [{status_tag}]")
        report_lines.append(f"**Prompt:** {item.prompt}")
        report_lines.append(f"**Tools Invoked:** {item.tool_calls_count}")
        report_lines.append(f"**Findings:**\n{item.summary}\n")

    return "\n".join(report_lines)


async def dispatch_parallel_subagents(subtasks: List[Dict[str, str]]) -> str:
    """
    Fans out execution to multiple subagents concurrently and gathers results.

    Args:
        subtasks: List of subtask parameter dictionaries with 'id' and 'prompt'.

    Returns:
        Formatted summary report string.
    """
    if not subtasks:
        return "Error: No subtasks provided to spawn_parallel_subagents"

    print(f"\033[34m[fan-out] launching {len(subtasks)} subagents in parallel...\033[0m")
    tasks = [
        run_single_subagent(
            task_id=str(item.get("id", f"subtask_{idx + 1}")),
            prompt=str(item.get("prompt", ""))
        )
        for idx, item in enumerate(subtasks)
    ]

    results: List[SubtaskResult] = await asyncio.gather(*tasks)
    print(f"\033[34m[fan-in] all {len(results)} subagents completed. Aggregating findings...\033[0m")

    return format_subagent_report(results)


# Tool definition for parallel subagent spawning
PARALLEL_SUBAGENT_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "spawn_parallel_subagents",
    "description": (
        "Spawns multiple subagents concurrently in isolated contexts using asyncio.gather. "
        "Ideal for parallel exploration, reviewing multiple files simultaneously, or testing "
        "different hypotheses at the same time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "description": "List of subtasks to execute in parallel.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short identifier for the subtask (e.g. 'check_auth', 'check_database')."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Explicit instructions for this subagent to execute."
                        }
                    },
                    "required": ["id", "prompt"]
                }
            }
        },
        "required": ["subtasks"]
    }
}

PARALLEL_TOOLS: List[Dict[str, Any]] = EXTENDED_TOOLS + [PARALLEL_SUBAGENT_TOOL_SCHEMA]


async def dispatch_lead_tool(block: Any) -> Tuple[str, str]:
    """
    Dispatches a single tool call for the lead agent.

    Args:
        block: The tool_use content block.

    Returns:
        Tuple of tool_use_id and string output.
    """
    tool_name = block.name
    tool_input = block.input

    if tool_name == "spawn_parallel_subagents":
        subtasks = tool_input.get("subtasks", [])
        output = await dispatch_parallel_subagents(subtasks)
        return block.id, output

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


async def lead_agent_loop(messages: List[Dict[str, Any]]) -> None:
    """
    Main autonomous thinking-acting loop for the Lead Agent with parallel subagents.

    Args:
        messages: Rolling conversation history.
    """
    while True:
        print("\n\033[36m> Lead Thinking...\033[0m")

        def _blocking_stream_call():
            # Streaming completion in background thread
            with client.messages.stream(
                model=MODEL,
                system=SYSTEM,
                messages=messages,
                tools=PARALLEL_TOOLS,
                max_tokens=8000,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                return stream.get_final_message()

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _blocking_stream_call)
        print()

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        tool_blocks = [
            b for b in response.content
            if getattr(b, "type", None) == "tool_use"
        ]

        execution_pairs = await asyncio.gather(
            *[dispatch_lead_tool(b) for b in tool_blocks]
        )
        turn_results = build_tool_results_list(tool_blocks, execution_pairs)
        messages.append({"role": "user", "content": turn_results})


async def main() -> None:
    """
    Main REPL entry point for Session 24.
    """
    print("\033[90ms24: parallel subagent spawning | fan-out exploration with asyncio.gather\033[0m\n")
    history: List[Dict[str, Any]] = []
    loop = asyncio.get_event_loop()

    while True:
        try:
            query: str = await loop.run_in_executor(
                None, lambda: input("\033[36ms24 >> \033[0m").strip()
            )
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session.")
            break

        if not query or query.lower() in ("q", "exit", "quit"):
            break

        history.append({"role": "user", "content": query})
        await lead_agent_loop(history)
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
