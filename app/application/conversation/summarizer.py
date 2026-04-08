from __future__ import annotations


def summarize_recent_messages(messages, recent_window: int = 4, summary_limit: int = 320) -> str:
    """Summarize recent messages into a bounded compact string for prompt context reuse."""

    recent = messages[-recent_window:]
    summary = " | ".join(f"{message.role}:{message.content[:60]}" for message in recent)
    return summary[:summary_limit]


def summarize_long_history(messages, keep_last: int = 6, summary_limit: int = 480) -> str:
    """Compress longer histories while preserving the most recent conversational turns."""

    if len(messages) <= keep_last:
        return summarize_recent_messages(messages, recent_window=keep_last, summary_limit=summary_limit)

    earlier = messages[:-keep_last]
    recent = messages[-keep_last:]
    earlier_summary = "; ".join(f"{message.role}:{message.content[:40]}" for message in earlier[-6:])
    recent_summary = " | ".join(f"{message.role}:{message.content[:60]}" for message in recent)
    combined = f"Earlier: {earlier_summary} || Recent: {recent_summary}"
    return combined[:summary_limit]
