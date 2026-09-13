"""stream_ollama_chat() のテスト（正常系・think フォールバック）。"""

import json
from io import BytesIO
from unittest.mock import patch, MagicMock, call

import pytest

from search import stream_ollama_chat, build_chat_payload


class TestBuildChatPayload:
    """build_chat_payload() のテスト。"""

    def test_basic_payload(self):
        body = build_chat_payload("gpt-oss:20b", "sys", "user")
        assert body["model"] == "gpt-oss:20b"
        assert body["stream"] is True
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        # think は常に明示送信する契約（F-7）。gpt-ossのreasoning未指定は最小のlow。
        assert body["think"] == "low"

    def test_payload_with_reasoning(self):
        body = build_chat_payload("gpt-oss:20b", "sys", "user", reasoning="high")
        assert body["think"] == "high"

    def test_payload_without_reasoning(self):
        body = build_chat_payload("gpt-oss:20b", "sys", "user", reasoning=None)
        assert body["think"] == "low"

    def test_payload_reasoning_off(self):
        body = build_chat_payload("gpt-oss:20b", "sys", "user", reasoning="off")
        assert body["think"] == "low"

    def test_payload_includes_keep_alive(self):
        body = build_chat_payload("gpt-oss:20b", "sys", "user")
        assert "keep_alive" in body


def _make_stream_response(chunks):
    """Ollama ストリーミングレスポンスを模擬する BytesIO を生成する。

    実 Ollama 仕様に合わせ、コンテンツは done=false の行で配信し、
    最後に content 空の done=true 終端行を付与する
    （iter_stream_chunks は done 行で break するため、content を done 行に載せない）。
    """
    lines = []
    for chunk in chunks:
        data = {"message": {"content": chunk}, "done": False}
        lines.append(json.dumps(data).encode("utf-8") + b"\n")
    lines.append(json.dumps({"message": {"content": ""}, "done": True}).encode("utf-8") + b"\n")
    return BytesIO(b"".join(lines))


class TestStreamOllamaChat:
    """stream_ollama_chat() のテスト。"""

    @patch("search.urllib.request.urlopen")
    def test_normal_streaming(self, mock_urlopen):
        mock_resp = _make_stream_response(["Hello", " World"])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = stream_ollama_chat("gpt-oss:20b", "sys", "question")
        assert result == "Hello World"

    @patch("search.urllib.request.urlopen")
    def test_reasoning_param_included(self, mock_urlopen):
        mock_resp = _make_stream_response(["OK"])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        stream_ollama_chat("gpt-oss:20b", "sys", "question", reasoning="high")

        # リクエストボディに think が含まれていることを確認
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["think"] == "high"

    @patch("search.urllib.request.urlopen")
    def test_think_fallback_on_http_error(self, mock_urlopen, capsys):
        from urllib.error import HTTPError

        # 1回目: think 付きで 400 エラー
        error_resp = MagicMock()
        error_resp.code = 400
        http_error = HTTPError(
            url="http://localhost:11434/api/chat",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=None,
        )

        # 2回目: think なしで成功
        success_resp = _make_stream_response(["Fallback OK"])
        success_resp.__enter__ = lambda s: s
        success_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [http_error, success_resp]

        result = stream_ollama_chat("gpt-oss:20b", "sys", "question", reasoning="high")
        assert result == "Fallback OK"

        # 2回呼ばれていること（1回目失敗 + 再試行）
        assert mock_urlopen.call_count == 2

        # 再試行リクエストに think が含まれていないこと
        retry_req = mock_urlopen.call_args_list[1][0][0]
        retry_body = json.loads(retry_req.data.decode("utf-8"))
        assert "think" not in retry_body

        # 警告出力を確認
        captured = capsys.readouterr()
        assert "think パラメータでエラー" in captured.err

    @patch("search.urllib.request.urlopen")
    def test_http_error_without_reasoning_raises(self, mock_urlopen):
        from urllib.error import HTTPError

        http_error = HTTPError(
            url="http://localhost:11434/api/chat",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )
        mock_urlopen.side_effect = http_error

        with pytest.raises(HTTPError):
            stream_ollama_chat("gpt-oss:20b", "sys", "question", reasoning=None)

    @patch("search.urllib.request.urlopen")
    def test_connection_error_handled(self, mock_urlopen):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection refused")

        result = stream_ollama_chat("gpt-oss:20b", "sys", "question")
        assert "接続エラー" in result

    @patch("search.urllib.request.urlopen")
    def test_timeout_handled(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()

        result = stream_ollama_chat("gpt-oss:20b", "sys", "question")
        assert "タイムアウト" in result


@pytest.mark.parametrize(
    "model", ["gpt-oss:20b", "gpt-oss:120b", "library/gpt-oss:20b", "GPT-OSS:20b"]
)
@pytest.mark.parametrize("reasoning", [None, "", "off"])
def test_gpt_oss_hidden_reasoning_uses_supported_minimum(model, reasoning):
    assert build_chat_payload(model, "sys", "user", reasoning=reasoning)["think"] == "low"


@pytest.mark.parametrize("model", ["qwen3.5:9b", "gpt-oss-custom:20b", "custom-model", "m"])
def test_other_models_keep_explicit_thinking_disable(model):
    assert build_chat_payload(model, "sys", "user", reasoning="off")["think"] is False


@pytest.mark.parametrize("reasoning", ["low", "medium", "high"])
def test_gpt_oss_explicit_reasoning_level_is_preserved(reasoning):
    assert (
        build_chat_payload("gpt-oss:20b", "sys", "user", reasoning=reasoning)["think"] == reasoning
    )
