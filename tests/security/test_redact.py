"""Tests for ``agent_scaffold._redact``.

The bar is "the original credential never appears in the output". Each
test case names the credential shape it targets so adding a new provider
is a one-line change.
"""

from __future__ import annotations

from agent_scaffold._redact import contains_secret_shape, redact, redact_obj


def test_anthropic_key_redacted() -> None:
    secret = "sk-ant-api03-AbCdEf123456789xYz"
    text = f"using key {secret} for the call"
    out = redact(text)
    assert secret not in out
    assert "sk-ant-...REDACTED" in out


def test_openai_style_key_redacted() -> None:
    secret = "sk-" + "A" * 40
    out = redact(f"key {secret}")
    assert secret not in out


def test_aws_access_key_id_redacted() -> None:
    secret = "AKIA" + "B" * 16
    out = redact(f"id={secret}")
    assert secret not in out


def test_postgres_url_credentials_redacted() -> None:
    url = "postgres://alice:s3cret-pw@db.example.com:5432/app"
    out = redact(f"DATABASE_URL={url}")
    assert "s3cret-pw" not in out
    assert "REDACTED@" in out


def test_redis_url_with_password_redacted() -> None:
    out = redact("connecting to redis://:topsecret@cache:6379/0")
    assert "topsecret" not in out
    assert "REDACTED@" in out


def test_bearer_token_redacted() -> None:
    out = redact("Authorization: Bearer abc123.token.xyz")
    assert "abc123.token.xyz" not in out


def test_github_pat_redacted() -> None:
    pat = "github_pat_" + "X" * 50
    out = redact(f"GITHUB_TOKEN={pat}")
    assert pat not in out


def test_slack_token_redacted() -> None:
    out = redact("token=xoxb-12345-abcdefghijklmn")
    assert "xoxb-12345-abcdefghijklmn" not in out


def test_non_secret_text_unchanged() -> None:
    text = "plain log line without any credentials"
    assert redact(text) == text


def test_empty_string_handled() -> None:
    assert redact("") == ""


def test_redact_obj_walks_dict_and_list() -> None:
    payload = {
        "error": "rejected by https://api.example.com",
        "key": "sk-ant-api03-secretsecretsecret",
        "history": [
            {"detail": "DATABASE_URL=postgres://u:pw@db/app"},
            {"detail": "ok"},
        ],
        "count": 3,  # non-string preserved
    }
    cleaned = redact_obj(payload)
    assert cleaned["count"] == 3
    assert "sk-ant-api03-secretsecretsecret" not in cleaned["key"]
    assert "pw" not in cleaned["history"][0]["detail"]
    assert cleaned["history"][1]["detail"] == "ok"


def test_contains_secret_shape_helper() -> None:
    assert contains_secret_shape("sk-ant-AAAAAAAAAA")
    assert not contains_secret_shape("nothing secret here")


def test_multiple_secrets_in_one_line_all_redacted() -> None:
    line = "ANTHROPIC_API_KEY=sk-ant-aaaaaaaaaaa AWS_KEY=AKIA" + "B" * 16
    out = redact(line)
    assert "sk-ant-aaaaaaaaaaa" not in out
    assert "AKIA" + "B" * 16 not in out
