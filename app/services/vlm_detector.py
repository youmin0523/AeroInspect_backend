# =============================================
# app/services/vlm_detector.py
# 역할: 비전 LLM(Gemini/Claude/GPT-4o) 기반 하자 검출 서비스
#       - 학습 ONNX 모델 검출률 저하(M4 mAP 0.503) 대안 / 병행 비교 PoC
#       - 이미지/키프레임을 LLM에 직접 보내 20종 하자 판정
#       - classify  : 이미지 단위 판정 (어떤 하자 + 심각도 + confidence, bbox 없음)
#       - grounding : Gemini normalized bbox(0~1000) 포함
#       - 프로바이더 추상화 (gemini 기본 · claude · openai)
#       - 비용 가드: 세마포어 동시성 제한, 일일 호출 상한, 프레임 해시 캐시
#
# 재사용:
#   - Gemini 호출 패턴: llm_report.py _stream_gemini
#   - 20종 taxonomy: utils/severity_mapper.py DEFECT_CATALOG / get_severity_by_code
#   - 응답 스키마: schemas/detection.py VLMDetection / VLMDetectionResult
# =============================================

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.schemas.detection import (
    ImageShape,
    VLMDetection,
    VLMDetectionResult,
)
from app.utils.severity_mapper import DEFECT_CATALOG, get_severity_by_code

logger = logging.getLogger(__name__)


class VLMDetector:
    """비전 LLM 하자 검출 서비스 (싱글톤)."""

    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(max(1, settings.VLM_MAX_CONCURRENCY))
        # 일일 호출 상한 카운터 (UTC 날짜 기준 리셋)
        self._call_count = 0
        self._call_day = self._today()
        # 카운터/캐시 동시 접근 보호 (check-then-act, day-rollover race 방지)
        self._counter_lock = asyncio.Lock()
        # 프레임 해시 캐시 (동일 프레임 중복 호출 차단). LRU 흉내 — 단순 dict + 상한.
        self._cache: Dict[str, VLMDetectionResult] = {}
        self._cache_order: List[str] = []
        self._cache_max = 256
        # ── 외부 LLM 클라이언트 캐시 (호출마다 새 httpx/SDK 생성 방지) ──
        self._anthropic_client = None
        self._openai_client = None
        self._gemini_configured = False

    # ── 외부 클라이언트 캐시 헬퍼 ─────────────────
    def _get_anthropic(self):
        if self._anthropic_client is None:
            import anthropic

            self._anthropic_client = anthropic.AsyncAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                timeout=settings.LLM_REQUEST_TIMEOUT,
            )
        return self._anthropic_client

    def _get_openai(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.LLM_REQUEST_TIMEOUT,
            )
        return self._openai_client

    def _get_gemini_model(self, model: str):
        """genai.configure 는 프로세스 전역 상태 — 1회만 호출(동시 race 방지)."""
        import google.generativeai as genai

        if not self._gemini_configured:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self._gemini_configured = True
        return genai.GenerativeModel(model)

    # ── 공개 API ──────────────────────────────
    async def detect(
        self,
        image_bytes: bytes,
        *,
        mode: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> VLMDetectionResult:
        """이미지 1장 → VLM 검출 결과.

        Args:
            image_bytes: 원본 이미지 바이트 (JPEG/PNG 등)
            mode: "classify" | "grounding" (None이면 settings.VLM_MODE)
            provider: "gemini" | "claude" | "openai" (None이면 settings.VLM_PROVIDER)
            model: 모델명 오버라이드 (None이면 settings.VLM_MODEL)
        """
        mode = (mode or settings.VLM_MODE).lower()
        provider = (provider or settings.VLM_PROVIDER).lower()
        model = model or settings.VLM_MODEL

        width, height = self._image_size(image_bytes)
        shape = ImageShape(width=width, height=height)

        # 캐시 조회 (provider/model/mode/이미지 동일 시 재사용)
        cache_key = self._cache_key(image_bytes, provider, model, mode)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"cached": True})

        await self._reserve_call_slot()

        prompt = self._build_prompt(mode)
        t0 = time.perf_counter()
        async with self._sem:
            try:
                if provider == "gemini":
                    raw = await self._call_gemini(prompt, image_bytes, model)
                elif provider == "claude":
                    raw = await self._call_claude(prompt, image_bytes, model)
                elif provider == "openai":
                    raw = await self._call_openai(prompt, image_bytes, model)
                else:
                    raise ValueError(f"알 수 없는 VLM_PROVIDER: {provider}")
            except BaseException:
                await self._refund_call_slot()  # 실패는 쿼터 환불
                raise
        latency_ms = (time.perf_counter() - t0) * 1000.0

        items = self._parse_items(raw)
        detections = self._to_detections(items, mode, width, height)

        result = VLMDetectionResult(
            detections=detections,
            has_defect=len(detections) > 0,
            defect_count=len(detections),
            provider=provider,
            model=model,
            mode=mode,
            latency_ms=round(latency_ms, 1),
            cached=False,
            image_shape=shape,
        )
        self._cache_put(cache_key, result)
        return result

    # ── 판정(adjudication) API: ONNX 후보 검증 + 누락 보완 ──
    async def adjudicate(
        self,
        image_bytes: bytes,
        candidates: List[Dict[str, Any]],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        conflict_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """ONNX 후보 리스트를 받아 LLM이 검증/교정/기각 + 누락 하자 보완.

        Args:
            candidates: [{"id":int, "class_name":str, "conf":float, "bbox_xyxy":[x1,y1,x2,y2]}]
            conflict_ids: 재판정(2차) 대상 후보 id. 주어지면 충돌 재심 프롬프트 사용.

        Returns:
            {"verdicts": [{"id", "verdict", "class_name", "conf", "reason"}],
             "missed":   [{"class_name", "conf", "box_2d", "reason"}]}
        """
        provider = (provider or settings.VLM_PROVIDER).lower()
        model = model or settings.VLM_MODEL

        await self._reserve_call_slot()
        prompt = self._build_adjudication_prompt(candidates, conflict_ids)

        async with self._sem:
            try:
                if provider == "gemini":
                    raw = await self._call_gemini(prompt, image_bytes, model)
                elif provider == "claude":
                    raw = await self._call_claude(prompt, image_bytes, model)
                elif provider == "openai":
                    raw = await self._call_openai(prompt, image_bytes, model)
                else:
                    raise ValueError(f"알 수 없는 VLM_PROVIDER: {provider}")
            except BaseException:
                await self._refund_call_slot()  # 실패는 쿼터 환불
                raise

        return self._parse_adjudication(raw)

    def _build_adjudication_prompt(
        self, candidates: List[Dict[str, Any]], conflict_ids: Optional[List[int]]
    ) -> str:
        """ONNX 후보 + 20종 taxonomy 주입 판정 프롬프트."""
        lines = []
        for code, info in DEFECT_CATALOG.items():
            lines.append(f"- class_name=\"{info['class_name']}\" | {info['name']} | 영역 {info['area']}")
        catalog = "\n".join(lines)

        cand_lines = []
        for c in candidates:
            bbox = c.get("bbox_xyxy") or []
            bstr = (
                f"bbox(px)=[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]"
                if len(bbox) == 4 else "위치 미상(이미지 전체)"
            )
            cand_lines.append(
                f"[{c['id']}] class_name=\"{c.get('class_name','')}\" "
                f"ONNX신뢰도={float(c.get('conf', 0)):.2f} {bstr}"
            )
        cand_block = "\n".join(cand_lines) if cand_lines else "(후보 없음)"

        conflict_note = ""
        if conflict_ids:
            conflict_note = (
                f"\n[재심 요청] 후보 {conflict_ids} 는 ONNX가 높은 신뢰도로 검출했으나 "
                "1차 판정과 충돌했습니다. 해당 박스 영역을 다시 면밀히 보고 최종 판정하세요.\n"
            )

        return f"""당신은 건축물 하자 점검 검증 전문가입니다. 자동 검출기(ONNX)가 제안한 하자 후보를
원본 이미지와 대조하여 검증하세요. 당신은 이미지 전체 맥락(표면 종류: 바닥/유리/벽/창틀 등)을
볼 수 있으므로, 검출기가 종류를 혼동한 경우 교정하는 것이 핵심 역할입니다.

[하자 분류 체계 — class_name 만 사용]
{catalog}

[ONNX 후보]
{cand_block}
{conflict_note}
[판정 규칙]
1. 각 후보에 대해 verdict 를 결정:
   - "confirm": 그 위치에 그 종류 하자가 실제로 있음.
   - "reclassify": 그 위치에 하자는 있으나 종류가 틀림 → class_name 을 올바른 것으로 교정.
     (예: 검출기가 바닥 표면 하자를 유리 스크래치로 오인 → 표면 맥락을 보고 교정)
   - "reject": 그 위치에 하자가 없음(정상/그림자/반사/사물 오인).
2. conf 는 당신의 확신도(0.0~1.0). 애매하면 낮추세요.
3. 후보에 없지만 당신이 명확히 보는 하자는 "missed" 에 추가 (class_name + box_2d[ymin,xmin,ymax,xmax] 0~1000 정규화).
   확실하지 않으면 추가하지 마세요(과검출 금지).
4. 목록 외 class_name 은 절대 사용 금지.

[출력 — 순수 JSON 만]
{{"verdicts": [{{"id": <후보 id>, "verdict": "confirm|reclassify|reject", "class_name": "<최종 class_name>", "conf": <0.0~1.0>, "reason": "<짧은 근거>"}}],
 "missed": [{{"class_name": "<class_name>", "conf": <0.0~1.0>, "box_2d": [ymin,xmin,ymax,xmax], "reason": "<근거>"}}]}}"""

    @staticmethod
    def _parse_adjudication(raw: str) -> Dict[str, Any]:
        """판정 응답 파싱 (verdicts + missed)."""
        if not raw:
            return {"verdicts": [], "missed": []}
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        data: Any = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = None
        if not isinstance(data, dict):
            logger.warning("VLM 판정 JSON 파싱 실패: %s", text[:200])
            return {"verdicts": [], "missed": []}
        verdicts = [v for v in data.get("verdicts", []) if isinstance(v, dict)]
        missed = [m for m in data.get("missed", []) if isinstance(m, dict)]
        return {"verdicts": verdicts, "missed": missed}

    # ── 프롬프트 ──────────────────────────────
    def _build_prompt(self, mode: str) -> str:
        """20종 taxonomy를 동적 주입 (하드코딩 금지)."""
        lines = []
        for code, info in DEFECT_CATALOG.items():
            lines.append(
                f"- {code} | class_name=\"{info['class_name']}\" | "
                f"{info['name']} | 영역 {info['area']} | 기본심각도 {info['severity']}"
            )
        catalog = "\n".join(lines)

        bbox_clause = (
            '각 항목에 "box_2d": [ymin, xmin, ymax, xmax] (0~1000 정규화 정수)를 포함하세요. '
            "박스는 하자가 보이는 영역을 최대한 좁게 감싸야 합니다."
            if mode == "grounding"
            else "이미지에 하자가 있는지(이미지 단위)만 판정하며 박스 좌표는 출력하지 않습니다."
        )

        return f"""당신은 건축물 입주 전 하자 점검 전문가입니다. 주어진 이미지를 분석해
아래 20종 하자 분류 체계에 해당하는 하자만 식별하세요.

[하자 분류 체계 — class_name 만 사용]
{catalog}

[판정 규칙]
1. 위 목록의 class_name 에 정확히 해당하는 하자만 보고합니다. 목록 외 항목은 무시합니다.
2. [마킹 무시] 점검자가 표시한 테이프·화살표·동그라미·스티커·손글씨·형광펜 자국은
   하자가 아닙니다. 절대 그 마킹 자체를 검출하지 말고, 위치 힌트로도 쓰지 마세요
   (실제 현장에는 마킹이 없는 게 정상입니다). 마킹이 가리키는 '실제 표면의 하자'만 봅니다.
3. [사물 vs 시공상태] '사물이 거기 있다'는 것 자체는 하자가 아닙니다(가구·가전·스위치 등의
   정상 표면·무늬·그림자·정상 이음매를 하자로 오인 금지). 단, 시공 요소(빌트인 가구·창호·
   문틀·몰딩·걸레받이)의 시공/정렬/부착 상태에 문제가 있으면 하자입니다 — 수직·수평도 불량,
   틀 직각도 불량, 들뜸·기울어짐·부착불량, 마감 틈·코킹 불량 등. '사물 자체'가 아니라
   그 '시공/마감 상태'를 판정하세요.
4. [표면 먼저] 먼저 부위의 표면을 판단하고(벽/천장/바닥/창유리/창틀/문틀) 그 표면에
   물리적으로 가능한 하자만 보고합니다. 예: 바닥의 선형 결함은 '유리 스크래치'가 아니라
   '타일 줄눈/바닥 균열'입니다. 유리 결함은 창유리에서만 봅니다.
5. [정밀/재현 균형] 안전·누수 직결(구조/마감 균열·방수/누수·단열)은 미탐 비용이 크니
   의심되면 낮은 conf 로라도 보고하고, 미관 하자(오염·도색얼룩·찍힘·스크래치)는 오탐 시
   불필요한 출장이 생기니 명확히 보일 때만 보고하세요(애매하면 conf 0.5 미만).
6. 정상 부위·그림자·반사·정상 시공 이음매·일반 텍스처는 하자가 아닙니다. 확신 없으면 보고 금지.
7. 동일 하자가 여러 곳이면 각각 별도 항목으로 보고합니다.
8. conf 는 0.0~1.0 사이 당신의 확신도입니다.
9. {bbox_clause}

[출력 형식 — 반드시 순수 JSON 만, 다른 텍스트 금지]
{{"detections": [
  {{"class_name": "<목록의 class_name>", "conf": <0.0~1.0>, "reasoning": "<짧은 근거>"{', "box_2d": [ymin,xmin,ymax,xmax]' if mode == 'grounding' else ''}}}
]}}
하자가 없으면 {{"detections": []}} 를 반환하세요."""

    # ── 프로바이더 호출 ───────────────────────
    async def _call_gemini(self, prompt: str, image_bytes: bytes, model: str) -> str:
        gm = self._get_gemini_model(model)
        parts = [
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes},
        ]
        resp = await asyncio.to_thread(
            gm.generate_content,
            parts,
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": settings.LLM_REQUEST_TIMEOUT},
        )
        return resp.text or "{}"

    async def _call_claude(self, prompt: str, image_bytes: bytes, model: str) -> str:
        client = self._get_anthropic()
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        msg = await client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        ) or "{}"

    async def _call_openai(self, prompt: str, image_bytes: bytes, model: str) -> str:
        client = self._get_openai()
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        return resp.choices[0].message.content or "{}"

    # ── 응답 파싱 ─────────────────────────────
    @staticmethod
    def _parse_items(raw: str) -> List[Dict[str, Any]]:
        """LLM 응답 텍스트 → detections 리스트. 견고한 JSON 추출."""
        if not raw:
            return []
        text = raw.strip()
        # 코드펜스 제거
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 본문 속 첫 JSON 객체만 추출 시도
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                logger.warning("VLM 응답 JSON 파싱 실패: %s", text[:200])
                return []
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.warning("VLM 응답 JSON 재파싱 실패: %s", text[:200])
                return []
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        items = data.get("detections", []) if isinstance(data, dict) else []
        return [d for d in items if isinstance(d, dict)]

    @staticmethod
    def _to_detections(
        items: List[Dict[str, Any]], mode: str, width: int, height: int
    ) -> List[VLMDetection]:
        out: List[VLMDetection] = []
        for it in items:
            class_name = str(it.get("class_name", "")).strip()
            if not class_name:
                continue
            info = get_severity_by_code(class_name)
            # 목록에 없는(X-00) 환각 클래스는 버림
            if info.get("code") == "X-00":
                logger.debug("VLM 목록 외 클래스 무시: %s", class_name)
                continue
            try:
                conf = float(it.get("conf", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))

            bbox: List[float] = []
            localization = "image_level"
            box = it.get("box_2d")
            if mode == "grounding" and isinstance(box, (list, tuple)) and len(box) == 4:
                bbox = VLMDetector._box2d_to_xyxy(box, width, height)
                localization = "bbox"
            else:
                bbox = [0.0, 0.0, float(width), float(height)]

            out.append(
                VLMDetection(
                    **{"class": info["class_name"]},
                    class_display_ko=info.get("name", ""),
                    code=info.get("code", ""),
                    area=info.get("area", ""),
                    conf=conf,
                    severity=info.get("severity"),
                    bbox_xyxy=bbox,
                    localization=localization,
                    reasoning=str(it.get("reasoning", ""))[:300],
                )
            )
        return out

    @staticmethod
    def _box2d_to_xyxy(box: Any, width: int, height: int) -> List[float]:
        """Gemini normalized [ymin,xmin,ymax,xmax] (0~1000) → 픽셀 [x1,y1,x2,y2]."""
        ymin, xmin, ymax, xmax = (float(v) for v in box)
        x1 = xmin / 1000.0 * width
        y1 = ymin / 1000.0 * height
        x2 = xmax / 1000.0 * width
        y2 = ymax / 1000.0 * height
        x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

    # ── 유틸: 이미지 크기 / 캐시 / 비용 가드 ──
    @staticmethod
    def _image_size(image_bytes: bytes) -> Tuple[int, int]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("이미지 디코딩 실패 (지원되지 않는 포맷).")
        h, w = frame.shape[:2]
        return int(w), int(h)

    @staticmethod
    def _cache_key(image_bytes: bytes, provider: str, model: str, mode: str) -> str:
        h = hashlib.md5(image_bytes).hexdigest()
        return f"{provider}:{model}:{mode}:{h}"

    def _cache_put(self, key: str, result: VLMDetectionResult) -> None:
        if key in self._cache:
            return
        self._cache[key] = result
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    async def _reserve_call_slot(self) -> None:
        """일일 상한 체크 + 카운터 증가를 원자적으로 수행 (check-then-act race 방지).

        호출 직전 슬롯을 선점(increment)하므로 동시성 하에서도 상한을 정확히 강제한다.
        실패 시에는 _refund_call_slot 으로 환불 — 실패한 호출이 쿼터를 소모하지 않게.
        """
        async with self._counter_lock:
            today = self._today()
            if today != self._call_day:
                self._call_day = today
                self._call_count = 0
            if self._call_count >= settings.VLM_DAILY_CALL_CAP:
                raise VLMQuotaExceeded(
                    f"VLM 일일 호출 상한 초과 ({settings.VLM_DAILY_CALL_CAP}). "
                    "VLM_DAILY_CALL_CAP 조정 또는 내일 재시도."
                )
            self._call_count += 1

    async def _refund_call_slot(self) -> None:
        """프로바이더 호출 실패 시 선점한 슬롯을 환불."""
        async with self._counter_lock:
            if self._call_count > 0:
                self._call_count -= 1

    # ── 상태 조회 (관측/디버깅용) ─────────────
    def stats(self) -> Dict[str, Any]:
        return {
            "call_count_today": self._call_count,
            "daily_cap": settings.VLM_DAILY_CALL_CAP,
            "cache_size": len(self._cache),
            "provider": settings.VLM_PROVIDER,
            "model": settings.VLM_MODEL,
            "mode": settings.VLM_MODE,
        }


class VLMQuotaExceeded(RuntimeError):
    """VLM 일일 호출 상한 초과."""


# 싱글톤 (pipeline / LLMReportService 패턴)
vlm_detector = VLMDetector()


async def detect_vlm_async(
    image_bytes: bytes,
    *,
    mode: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> VLMDetectionResult:
    """공개 API — VLM 검출 (async)."""
    return await vlm_detector.detect(
        image_bytes, mode=mode, provider=provider, model=model
    )
