def enqueue(task_name: str, payload: dict) -> dict:
    return {"task": task_name, "payload": payload, "status": "queued"}
