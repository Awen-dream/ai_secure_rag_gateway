import unittest

from app.main import app


class AppImportTest(unittest.TestCase):
    def test_app_import(self) -> None:
        self.assertEqual(app.title, "Secure Enterprise RAG Gateway")


if __name__ == "__main__":
    unittest.main()
