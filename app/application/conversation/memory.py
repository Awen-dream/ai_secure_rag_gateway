from app.infrastructure.db.repositories.sqlite import SQLiteRepository


def get_session_messages(repository: SQLiteRepository, session_id: str):
    return repository.list_messages(session_id)
