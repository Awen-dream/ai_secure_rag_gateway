from app.main import app


def test_app_import() -> None:
    assert app.title == "Secure Enterprise RAG Gateway"
