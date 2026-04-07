import unittest
from unittest.mock import patch

from app.application.ingestion import tokenization


class _FakeTokenizer:
    def encode(self, text: str, *, disallowed_special: tuple[str, ...] = ()) -> list[int]:
        return [101, 102, 103, len(text)]


class TokenizationTests(unittest.TestCase):
    def test_estimate_token_count_falls_back_without_tokenizer(self) -> None:
        with patch.object(tokenization, "_get_tokenizer", return_value=None):
            count = tokenization.estimate_token_count("这是一个中文段落 mixed with English.")

        self.assertGreater(count, 0)

    def test_estimate_token_count_uses_real_tokenizer_when_available(self) -> None:
        with patch.object(tokenization, "_get_tokenizer", return_value=_FakeTokenizer()):
            count = tokenization.estimate_token_count("security boundary")

        self.assertEqual(count, 4)

    def test_encoding_name_overrides_model_lookup(self) -> None:
        tokenization._get_tokenizer.cache_clear()
        with patch.object(tokenization, "tiktoken", create=True) as fake_tiktoken:
            fake_tiktoken.get_encoding.return_value = _FakeTokenizer()
            count = tokenization.estimate_token_count("tenant scope", encoding_name="cl100k_base")

        self.assertEqual(count, 4)
        fake_tiktoken.get_encoding.assert_called_once_with("cl100k_base")


if __name__ == "__main__":
    unittest.main()
