"""Roboflow 호스팅 모델 로딩 + 경기장 좌표계 구성.

가중치 .pt 파일은 공개 배포되지 않으므로 `inference` 패키지로 Roboflow에서 받아 쓴다.
(Roboflow 무료 계정의 API 키 필요)
"""

from dataclasses import dataclass
from typing import Any

# football-players-detection-3zvbc 의 클래스 ID
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

CLASS_NAMES = {BALL_ID: "ball", GOALKEEPER_ID: "goalkeeper",
               PLAYER_ID: "player", REFEREE_ID: "referee"}

PLAYER_MODEL_ID = "football-players-detection-3zvbc/11"
FIELD_MODEL_ID = "football-field-detection-f07vi/14"


@dataclass
class Models:
    player: Any   # 선수·GK·심판·공 탐지
    field: Any    # 경기장 키포인트 32개


def load(api_key: str) -> Models:
    """Roboflow 모델 두 개를 로드한다. 최초 호출 시 가중치를 내려받아 캐시한다."""
    if not api_key:
        raise ValueError(
            "Roboflow API 키가 비어 있다. "
            "https://app.roboflow.com/settings/api 에서 발급받아 전달할 것."
        )
    from inference import get_model  # 무거운 임포트라 지연시킨다

    return Models(
        player=get_model(model_id=PLAYER_MODEL_ID, api_key=api_key),
        field=get_model(model_id=FIELD_MODEL_ID, api_key=api_key),
    )


def pitch_config(preset):
    """프리셋의 실측 규격을 반영한 SoccerPitchConfiguration.

    Roboflow 기본값은 120x70m(규정 최대치)라 그대로 쓰면 거리·속도가 부풀려진다.
    페널티박스 등 고정 규격은 그대로 두고 전체 길이·폭만 실측값으로 바꾼다.
    좌표 단위는 cm이므로 사용처에서 /100 하여 미터로 쓴다.
    """
    from sports.configs.soccer import SoccerPitchConfiguration

    return SoccerPitchConfiguration(
        length=preset.pitch_length_cm,
        width=preset.pitch_width_cm,
    )
