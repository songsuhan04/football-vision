"""모델 가중치 로딩 + 경기장 좌표계 구성.

Roboflow 의 `inference` 패키지는 Python <3.13 만 지원한다.
Colab 이 3.13 으로 올라가면서 설치가 불가능해졌으므로,
가중치(.pt)를 직접 받아 `ultralytics` 로 구동한다. Roboflow API 키도 필요 없다.

가중치는 roboflow/sports 예제가 쓰는 것과 같은 파일명·같은 베이스(yolov8l / yolov8l-pose)의
HuggingFace 미러에서 받는다. 구조는 로드 시점에 검증한다 (아래 _verify_*).
"""

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# (HF repo, 파일명)
WEIGHTS: Dict[str, Tuple[str, str]] = {
    "player": ("Sabkat/football-player-detection", "football-player-detection.pt"),
    "field": ("Sabkat/football-pitch-detection", "football-pitch-detection.pt"),
}

# 선수 탐지 모델이 내야 하는 클래스. ID 는 하드코딩하지 않고 이름으로 찾는다.
EXPECTED_CLASSES = ("ball", "goalkeeper", "player", "referee")

# 경기장 키포인트 개수. SoccerPitchConfiguration.vertices 와 일치해야 한다.
EXPECTED_KEYPOINTS = 32

DEFAULT_CACHE = "weights"


@dataclass
class Models:
    player: Any
    field: Any
    cls: Dict[str, int] = field(default_factory=dict)   # 이름 → 클래스 ID
    n_keypoints: int = 0
    device: str = "cpu"

    def id_of(self, name: str) -> int:
        return self.cls[name]


def check_env() -> None:
    """필요한 패키지가 다 있는지 미리 확인한다.

    모델 로드 단계에서야 터지면 원인을 찾기 어렵다. 앞단에서 한 번에 알려준다.
    """
    import importlib

    missing = []
    for mod, hint in (("ultralytics", "ultralytics"),
                      ("supervision", "supervision"),
                      ("sports", "git+https://github.com/roboflow/sports.git")):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append((mod, hint))
    if missing:
        pkgs = " ".join(h for _, h in missing)
        raise ImportError(
            "누락된 패키지: " + ", ".join(m for m, _ in missing) +
            f" — 노트북 1번 셀을 실행할 것. 직접 설치: !pip install -q {pkgs}"
        )


def _download(repo: str, fname: str, cache_dir: str) -> str:
    """HuggingFace 에서 가중치를 받는다. 이미 있으면 건너뛴다."""
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, fname)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        return dest

    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    tmp = dest + ".part"

    # 매 블록마다 찍으면 로그가 폭주한다. 10% 단위로만 알린다.
    state = {"last": -1}

    def hook(blocks, block_size, total):
        if total <= 0:
            return
        pct = min(100, blocks * block_size * 100 // total)
        if pct >= state["last"] + 10:
            state["last"] = pct - (pct % 10)
            print(f"    {state['last']:3d}%", flush=True)

    print(f"  다운로드: {repo}/{fname}")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print("    완료")
    os.replace(tmp, dest)
    return dest


def _verify_player(model) -> Dict[str, int]:
    """선수 탐지 모델의 클래스를 확인하고 이름 → ID 매핑을 만든다."""
    names = model.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    lookup = {str(v).lower(): int(k) for k, v in names.items()}

    missing = [c for c in EXPECTED_CLASSES if c not in lookup]
    if missing:
        raise RuntimeError(
            f"선수 탐지 모델의 클래스가 예상과 다르다. 없는 클래스: {missing} / "
            f"모델이 가진 것: {sorted(lookup)}"
        )
    return {c: lookup[c] for c in EXPECTED_CLASSES}


def _verify_field(model) -> int:
    """경기장 모델이 32개 키포인트를 내는지 확인한다.

    개수가 다르면 SoccerPitchConfiguration.vertices 와 짝이 맞지 않아
    호모그래피가 엉뚱한 결과를 낸다.
    """
    shape = getattr(getattr(model, "model", None), "kpt_shape", None)
    if shape is None:
        raise RuntimeError("경기장 모델에 kpt_shape 가 없다. pose 모델이 아닌 것 같다.")
    n = int(shape[0])
    if n != EXPECTED_KEYPOINTS:
        raise RuntimeError(
            f"경기장 키포인트가 {n}개다. {EXPECTED_KEYPOINTS}개를 기대했다 — "
            "SoccerPitchConfiguration.vertices 와 짝이 맞지 않는다."
        )
    return n


def load(cache_dir: str = DEFAULT_CACHE, device: Optional[str] = None,
         weights: Optional[Dict[str, Tuple[str, str]]] = None) -> Models:
    """가중치를 받아 두 모델을 로드하고 구조를 검증한다."""
    check_env()
    from ultralytics import YOLO
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    src = weights or WEIGHTS
    player_pt = _download(*src["player"], cache_dir=cache_dir)
    field_pt = _download(*src["field"], cache_dir=cache_dir)

    player = YOLO(player_pt).to(device)
    fieldm = YOLO(field_pt).to(device)

    cls = _verify_player(player)
    n_kp = _verify_field(fieldm)

    return Models(player=player, field=fieldm, cls=cls, n_keypoints=n_kp, device=device)


def pitch_config(preset):
    """프리셋의 실측 규격을 반영한 SoccerPitchConfiguration.

    Roboflow 기본값은 120x70m(규정 최대치)라 그대로 쓰면 거리·속도가 부풀려진다.
    페널티박스 등 고정 규격은 그대로 두고 전체 길이·폭만 실측값으로 바꾼다.
    좌표 단위는 cm 이므로 사용처에서 /100 하여 미터로 쓴다.
    """
    from sports.configs.soccer import SoccerPitchConfiguration

    return SoccerPitchConfiguration(
        length=preset.pitch_length_cm,
        width=preset.pitch_width_cm,
    )
