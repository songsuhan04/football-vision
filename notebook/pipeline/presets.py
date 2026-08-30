"""소스별 파이프라인 파라미터 프리셋.

임계값은 **여기에만** 둔다. 다른 모듈에 하드코딩하지 않는다.
Stage 1(프로 중계) → Stage 2(아마추어 직촬) 전환이 프리셋 교체만으로 끝나야 하기 때문이다.
"""

from dataclasses import dataclass, replace
from typing import Dict


@dataclass(frozen=True)
class Preset:
    name: str

    # ---- 샘플링 ----
    scan_fps: float = 2.0       # 사전 스캔 (싸게 훑기)
    analyze_fps: float = 10.0   # 정밀 처리 (채택 구간에만)

    # ---- 선수 탐지 ----
    det_conf: float = 0.30
    nms_threshold: float = 0.50

    # ---- 경기장 키포인트 ----
    kp_conf: float = 0.50
    min_keypoints: int = 4      # 호모그래피 계산 최소 대응점

    # ---- 분석 가능 구간 판정 ----
    min_players: int = 6        # 클로즈업·관중샷을 걸러내는 가장 강력한 필터
    min_window_s: float = 8.0
    bridge_bad_frames: int = 1
    """구간 안에서 이어붙일 수 있는 연속 결손 프레임 수.

    스캔 한 프레임이 튀었다고 30초짜리 좋은 구간을 둘로 쪼개면 손해가 크다.
    이 길이 이하의 결손은 앞뒤가 정상일 때 메워서 구간을 잇는다 (나중에 보간 처리).
    """
    max_camera_speed: float = 25.0
    """프레임 간 카메라 이동 허용치 [m/s].

    호모그래피로 화면 기준점을 경기장 좌표로 옮겨 프레임 간 이동량을 재고,
    이를 시간으로 나눈 값. 초과하면 컷 또는 급격한 팬으로 보고 구간을 끊는다.
    """

    # ---- 호모그래피 스무딩 ----
    h_smooth_window: int = 5    # 홀수 권장. 클수록 떨림은 줄지만 빠른 팬에 늦게 반응

    # ---- 경기장 실측 규격 [cm] ----
    # Roboflow 기본 설정은 120x70m(최대 규격)이라 거리 지표가 부풀려진다.
    # 실제 경기장 값을 넣어야 이동거리·속도가 맞는다.
    pitch_length_cm: int = 10500
    pitch_width_cm: int = 6800

    # ---- 히트맵 격자 ----
    grid_cols: int = 12         # 105/12 = 8.75m
    grid_rows: int = 8          # 68/8  = 8.5m  → 거의 정사각 바둑판

    def with_(self, **kwargs) -> "Preset":
        """일부 값만 바꾼 사본. 실측 규격 주입 등에 사용."""
        return replace(self, **kwargs)


BROADCAST = Preset(
    name="broadcast",
    # 중계는 모델 학습 도메인이라 기본값이 잘 맞는다.
    # 컷이 잦으므로 카메라 속도 임계값이 컷 감지 역할을 겸한다.
    det_conf=0.30,
    kp_conf=0.50,
    min_players=6,
    h_smooth_window=5,
    bridge_bad_frames=1,
)

HANDHELD_ELEVATED = Preset(
    name="handheld_elevated",
    # 스마트폰 높은 위치 pitch view: 컷은 없지만 화각이 좁고 손떨림이 있다.
    det_conf=0.20,          # 먼 쪽 작은 선수를 놓치지 않도록 하향
    kp_conf=0.40,           # 라인이 흐려 키포인트 신뢰도가 낮게 나온다
    min_players=6,
    h_smooth_window=11,     # 손떨림 → 스무딩 강화
    max_camera_speed=40.0,  # 컷이 없으므로 끊을 이유가 적다. 느슨하게
    bridge_bad_frames=2,    # 흔들림 탓에 키포인트가 순간적으로 빠지는 일이 잦다
    # 경기장 실측값은 사용할 때 .with_(pitch_length_cm=..., pitch_width_cm=...) 로 주입
)

PRESETS: Dict[str, Preset] = {p.name: p for p in (BROADCAST, HANDHELD_ELEVATED)}


def get(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"알 수 없는 프리셋: {name!r}. 사용 가능: {list(PRESETS)}")
    return PRESETS[name]
