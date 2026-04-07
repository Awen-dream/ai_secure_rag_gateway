from app.infrastructure.db.repositories.base import MetadataRepository


def get_session_messages(repository: MetadataRepository, session_id: str):
    return repository.list_messages(session_id)
