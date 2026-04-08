from app.api.deps import get_document_ingestion_worker


def main() -> None:
    worker = get_document_ingestion_worker()
    worker.run_forever()


if __name__ == "__main__":
    main()
