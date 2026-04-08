from app.api.deps import get_feishu_source_sync_service
from app.core.config import settings


def main() -> None:
    service = get_feishu_source_sync_service()
    service.run_scheduler_forever(settings.source_sync_scheduler_poll_seconds)


if __name__ == "__main__":
    main()
