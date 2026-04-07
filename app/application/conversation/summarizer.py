def summarize_recent_messages(messages) -> str:
    recent = messages[-4:]
    return " | ".join(f"{message.role}:{message.content[:60]}" for message in recent)
