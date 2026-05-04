# =============================================
# tests/test_logging_json.py
# 역할: LOG_JSON=true 출력 유효성 검증
#       - configure_logging(json_output=True) 상태에서 로그 1줄이
#         유효한 JSON 이어야 함
#       - request_id contextvar가 자동 바인딩되어 같은 줄에 포함돼야 함
#       - LOG_JSON=false (콘솔) 출력은 JSON 파싱 실패해야 함 (색상 escape 포함)
#
# 실행: pytest tests/test_logging_json.py -v
# =============================================

from __future__ import annotations

import json

import pytest
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.core.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def reset_structlog():
    """각 테스트 후 structlog 설정/contextvars 초기화 — 다른 테스트에 오염 방지."""
    yield
    clear_contextvars()
    structlog.reset_defaults()
    # 다른 테스트가 기본 configure 상태를 기대할 수 있으므로 원복
    configure_logging(json_output=False, level="INFO")


def _emit_log_line(json_output: bool, caplog, **extra_context) -> str:
    """
    configure_logging 후 info 한 줄 찍고 caplog 로 최종 렌더링된 메시지 회수.
    structlog → stdlib logging 경로이므로 LogRecord.msg 가 곧 렌더링된 라인.

    extra_context: bind_contextvars 로 주입할 키/값
    """
    import logging as _logging

    if extra_context:
        bind_contextvars(**extra_context)

    # caplog 은 프로세스 전체 로그를 잡지만 레벨은 명시 필요
    caplog.clear()
    caplog.set_level(_logging.INFO)

    configure_logging(json_output=json_output, level="INFO")
    logger = get_logger("test")
    logger.info("http.request", status=200, duration_ms=12.3)

    assert caplog.records, "로그가 stdlib logging 으로 흐르지 않음"
    # 마지막 레코드의 메시지 = structlog 가 렌더링한 최종 문자열
    return caplog.records[-1].getMessage()


class TestJSONOutput:
    def test_json_output_is_parseable(self, caplog):
        line = _emit_log_line(json_output=True, caplog=caplog)
        parsed = json.loads(line)
        assert parsed["event"] == "http.request"
        assert parsed["status"] == 200
        assert parsed["duration_ms"] == pytest.approx(12.3)
        assert parsed["level"] == "info"
        assert "timestamp" in parsed

    def test_json_output_includes_bound_contextvars(self, caplog):
        """request_id 같은 contextvars 가 자동으로 이벤트에 병합돼야 함."""
        line = _emit_log_line(
            json_output=True,
            caplog=caplog,
            request_id="abc123",
            path="/api/v1/defects",
        )
        parsed = json.loads(line)
        assert parsed["request_id"] == "abc123"
        assert parsed["path"] == "/api/v1/defects"


class TestConsoleOutput:
    def test_console_output_is_not_json(self, caplog):
        """LOG_JSON=false (개발) 시에는 JSON 파싱 실패해야 함 (구분 보장)."""
        line = _emit_log_line(json_output=False, caplog=caplog)
        with pytest.raises(json.JSONDecodeError):
            json.loads(line)
        # event 이름은 그래도 평문으로 들어있음
        assert "http.request" in line
