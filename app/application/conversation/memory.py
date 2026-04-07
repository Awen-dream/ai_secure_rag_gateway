from app.infrastructure.db.repositories.memory import store


def get_session_messages(session_id: str):
    return store.chat_messages.get(session_id, [])
