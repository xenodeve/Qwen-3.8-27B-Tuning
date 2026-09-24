"""Image parts are rejected up front, before any GPU work (issue #85).

SUPERSEDES #84. The #84 substitution ([image omitted] + full inference) is
replaced by llama-server's behavior on a model with no vision tower: HTTP
500 `image input is not supported`, recorded in
launchers/serve-dual-nvfp4.bat. Rejecting is cheaper than inferring on a
note, and it is what our other profiles already do. The Anthropic path is
untouched -- its translator already turns images into text before the inner
endpoint ever sees them.
"""
import os
import sys

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.dirname(BENCH)
sys.path.insert(0, os.path.join(TUNING, "serving", "exl3"))
import server as exl3_server  # noqa: E402


def image_body():
    return {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }],
    }


def test_image_parts_are_rejected_before_any_inference():
    req, err = exl3_server.parse_request(image_body())
    assert req is None
    assert err == exl3_server.IMAGE_UNSUPPORTED == "image input is not supported"


def test_text_only_and_other_roles_pass_through():
    req, err = exl3_server.parse_request(
        {"messages": [{"role": "user", "content": "plain"}]})
    assert err is None and req["messages"] == [{"role": "user", "content": "plain"}]
    req, err = exl3_server.parse_request(
        {"messages": [{"role": "tool", "content": "ok"}]})
    assert err is None
    req, err = exl3_server.parse_request(
        {"messages": [{"role": "user",
                       "content": [{"type": "text", "text": "hi"}]}]})
    assert err is None


def test_the_rejection_maps_to_500_while_other_parse_errors_stay_400():
    server = open(os.path.join(TUNING, "serving", "exl3", "server.py"),
                  encoding = "utf-8").read()
    assert "err == IMAGE_UNSUPPORTED" in server
    assert "status = 500 if err == IMAGE_UNSUPPORTED else 400" in server


def test_normalize_messages_no_longer_substitutes_images():
    server = open(os.path.join(TUNING, "serving", "exl3", "server.py"),
                  encoding = "utf-8").read()
    assert "[image omitted]" not in server
    parts = [{"type": "text", "text": "hi"},
             {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]
    out = exl3_server.normalize_messages([{"role": "user", "content": parts}])
    assert out[0]["content"] == parts
