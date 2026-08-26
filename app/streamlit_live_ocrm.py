import streamlit as st
import cv2
import re
import time
import requests
import math

from pathlib import Path
from collections import Counter, deque
from threading import Lock

from ultralytics import YOLO
from paddlex.inference import create_predictor

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)

from streamlit_autorefresh import st_autorefresh


# ============================================================
# CONFIGURATION
# ============================================================

# Project root = one level up from this file (app/..).
BASE_DIR = Path(__file__).resolve().parent.parent

# Ships in models/best.pt (see README). Built with Path so it
# also works on Linux/Mac, not just Windows.
YOLO_PATH = str(BASE_DIR / "models" / "best.pt")

OCRM_URL = "http://127.0.0.1:8000"


# ============================================================
# PERFORMANCE
# ============================================================

YOLO_FRAME_STEP = 3

OCR_FRAME_STEP = 3

MIN_YOLO_CONF = 0.30

MIN_OCR_CONF = 0.50

CONFIRM_VOTES = 3

MAX_READINGS = 12

# posle 5 YOLO-проверок без подходящего ГРНЗ
# возвращаемся в SEARCH.
LOST_LIMIT = 5

# Number of stable YOLO detections required
# before starting OCR.
DETECTION_VOTES = 3

# Maximum allowed movement of detection center
# between consecutive YOLO detections.
DETECTION_CENTER_DISTANCE = 0.35


# ============================================================
# DETECTION TRACKING
# ============================================================

# Минимальный aspect ratio для объекта,
# похожего на автомобильный номер.
MIN_PLATE_ASPECT = 1.5
MAX_PLATE_ASPECT = 8.0

# Минимальная ширина detection.
MIN_PLATE_WIDTH = 35

# Минимальная высота detection.
MIN_PLATE_HEIGHT = 10

# Когда уже есть CONFIRMED:
# если новый detection находится слишком далеко
# от предыдущего bbox несколько раз подряд,
# считаем, что появилась другая машина.
CONFIRMED_IOU_THRESHOLD = 0.15

CONFIRMED_DIFFERENT_LIMIT = 2


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="KZ LPR + OCRM",
    page_icon="🚗",
    layout="wide",
)

st.title("🇰🇿 KZ License Plate + OCRM")

st.caption(
    "Realtime license plate recognition → "
    "OCRM lookup → vehicle information → visit"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    print("Loading YOLO...")

    model = YOLO(YOLO_PATH)

    print("Loading PaddleOCR...")

    ocr = create_predictor(
        "en_PP-OCRv5_mobile_rec",
        device="cpu",
    )

    print("Models loaded.")

    return model, ocr


with st.spinner("Loading YOLO and PaddleOCR..."):
    model, ocr = load_models()


# ============================================================
# SHARED STATE
# ============================================================

class LPRState:

    def __init__(self):

        self.lock = Lock()

        # SEARCH
        # RECOGNIZE
        # CONFIRMED
        self.state = "SEARCH"

        self.frame_number = 0

        self.yolo_calls = 0
        self.ocr_calls = 0

        self.vehicle_count = 0

        # Current confirmed plate.
        self.current_plate = ""

        self.current_confidence = 0.0

        # Latest OCR result.
        self.last_ocr = ""
        self.last_ocr_conf = 0.0

        # OCR readings for current vehicle.
        self.readings = deque(
            maxlen=MAX_READINGS
        )

        # Confirmed plates history.
        self.results = []

        # ----------------------------------------------------
        # Detection tracking
        # ----------------------------------------------------

        self.lost_count = 0

        self.last_detection = None

        # ----------------------------------------------------
        # SEARCH detection stabilization
        # ----------------------------------------------------

        self.detection_candidate = None
        self.detection_votes = 0

        # Detection belonging to confirmed vehicle.
        self.confirmed_detection = None

        # Number of consecutive detections
        # that look like a different object.
        self.different_detection_count = 0

        # ----------------------------------------------------
        # OCRM
        # ----------------------------------------------------

        self.ocrm_data = None

        self.ocrm_plate = ""

        self.ocrm_error = ""

        # ----------------------------------------------------
        # OCR status
        # ----------------------------------------------------

        self.ocr_failure_message = ""

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.processing_times = deque(
            maxlen=30
        )

        self.fps = 0.0


@st.cache_resource
def get_shared_state():

    return LPRState()


shared_state = get_shared_state()


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).upper().strip()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )


def valid_kz_plate(text):

    if not text:
        return False

    text = clean_text(text)

    if not 6 <= len(text) <= 12:
        return False

    if not re.fullmatch(
        r"[A-Z0-9]+",
        text,
    ):
        return False

    if not re.search(
        r"[A-Z]",
        text,
    ):
        return False

    if not re.search(
        r"[0-9]",
        text,
    ):
        return False

    return True


# ============================================================
# OCR NORMALIZATION
# ============================================================

def normalize_plate_candidate(text):

    text = clean_text(text)

    if not text:
        return ""

    if len(text) != 8:
        return text

    chars = list(text)

    digit_map = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "T": "7",
        "B": "8",
    }

    letter_map = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    }

    for index in [0, 1, 2, 6, 7]:

        if chars[index] in digit_map:

            chars[index] = digit_map[
                chars[index]
            ]

    for index in [3, 4, 5]:

        if chars[index] in letter_map:

            chars[index] = letter_map[
                chars[index]
            ]

    return "".join(chars)


# ============================================================
# CHARACTER VOTING
# ============================================================

def character_vote(readings):

    valid = []

    for reading in readings:

        candidate = normalize_plate_candidate(
            reading
        )

        if valid_kz_plate(candidate):

            valid.append(candidate)

    if not valid:

        return "", 0.0, 0

    counts = Counter(valid)

    most_common = counts.most_common(1)

    if not most_common:

        return "", 0.0, 0

    plate, exact_votes = most_common[0]

    position_confidences = []

    compatible = [
        value
        for value in valid
        if len(value) == len(plate)
    ]

    if compatible:

        for position in range(len(plate)):

            chars = [
                value[position]
                for value in compatible
            ]

            position_counts = Counter(chars)

            _, votes = (
                position_counts
                .most_common(1)[0]
            )

            position_confidences.append(
                votes / len(compatible)
            )

    if position_confidences:

        confidence = (
            sum(position_confidences)
            /
            len(position_confidences)
        )

    else:

        confidence = (
            exact_votes / len(valid)
        )

    return (
        plate,
        confidence,
        exact_votes,
    )


# ============================================================
# DETECTION GEOMETRY
# ============================================================

def detection_size(detection):

    if detection is None:
        return 0, 0

    x1, y1, x2, y2, _ = detection

    width = max(
        0,
        x2 - x1
    )

    height = max(
        0,
        y2 - y1
    )

    return width, height


def is_plate_detection(detection):

    """
    dop.filter YOLO.

    не принимаем любой bbox.
    Detection должен быть визуально похож
    на горизонтальный ГРНЗ
    """

    if detection is None:

        return False

    x1, y1, x2, y2, confidence = (
        detection
    )

    width = x2 - x1
    height = y2 - y1

    if width < MIN_PLATE_WIDTH:
        return False

    if height < MIN_PLATE_HEIGHT:
        return False

    if height <= 0:
        return False

    aspect = width / height

    if aspect < MIN_PLATE_ASPECT:
        return False

    if aspect > MAX_PLATE_ASPECT:
        return False

    if confidence < MIN_YOLO_CONF:
        return False

    return True


# ============================================================
# IOU
# ============================================================

def detection_centers_close(
    detection_a,
    detection_b,
):
    """
    Checks whether two YOLO detections
    are approximately in the same location.
    """

    if (
        detection_a is None
        or
        detection_b is None
    ):
        return False

    ax1, ay1, ax2, ay2, _ = detection_a
    bx1, by1, bx2, by2, _ = detection_b

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2

    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)

    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    reference_size = max(
        aw,
        ah,
        bw,
        bh,
    )

    distance = (
        (acx - bcx) ** 2
        +
        (acy - bcy) ** 2
    ) ** 0.5

    return (
        distance
        <=
        reference_size
        * DETECTION_CENTER_DISTANCE
    )

def calculate_iou(
    box_a,
    box_b,
):

    if box_a is None or box_b is None:

        return 0.0

    ax1, ay1, ax2, ay2, _ = box_a

    bx1, by1, bx2, by2, _ = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(
        0,
        ix2 - ix1
    )

    intersection_height = max(
        0,
        iy2 - iy1
    )

    intersection = (
        intersection_width
        *
        intersection_height
    )

    area_a = max(
        0,
        ax2 - ax1
    ) * max(
        0,
        ay2 - ay1
    )

    area_b = max(
        0,
        bx2 - bx1
    ) * max(
        0,
        by2 - by1
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:

        return 0.0

    return intersection / union


# ============================================================
# DETECTION SIMILARITY
# ============================================================

def detections_belong_to_same_plate(
    old_detection,
    new_detection,
):

    if (
        old_detection is None
        or
        new_detection is None
    ):

        return False

    iou = calculate_iou(
        old_detection,
        new_detection,
    )

    if iou >= CONFIRMED_IOU_THRESHOLD:

        return True

    # --------------------------------------------------------
    # Center-distance fallback.
    #
    # This is important because the camera may move slightly,
    # causing IoU to become small even for the same plate.
    # --------------------------------------------------------

    ox1, oy1, ox2, oy2, _ = old_detection

    nx1, ny1, nx2, ny2, _ = new_detection

    old_cx = (
        ox1 + ox2
    ) / 2

    old_cy = (
        oy1 + oy2
    ) / 2

    new_cx = (
        nx1 + nx2
    ) / 2

    new_cy = (
        ny1 + ny2
    ) / 2

    old_w = max(
        1,
        ox2 - ox1
    )

    old_h = max(
        1,
        oy2 - oy1
    )

    diagonal = math.sqrt(
        old_w ** 2
        +
        old_h ** 2
    )

    distance = math.sqrt(
        (old_cx - new_cx) ** 2
        +
        (old_cy - new_cy) ** 2
    )

    # If centers are still reasonably close,
    # treat as same plate.
    if distance <= diagonal * 1.5:

        return True

    return False


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_plate(frame):

    try:

        results = model(
            frame,
            imgsz=512,
            verbose=False,
        )

    except Exception as exc:

        print(
            f"YOLO error: {exc}"
        )

        return None

    with shared_state.lock:

        shared_state.yolo_calls += 1

    if not results:

        return None

    boxes = results[0].boxes

    if boxes is None:

        return None

    if len(boxes) == 0:

        return None

    # --------------------------------------------------------
    # Do NOT blindly take the first detection.
    #
    # Search through all detections and choose the best
    # detection that looks like a license plate.
    # --------------------------------------------------------

    candidates = []

    for index in range(len(boxes)):

        box = boxes[index]

        confidence = float(
            box.conf[0]
        )

        if confidence < MIN_YOLO_CONF:

            continue

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        detection = (
            x1,
            y1,
            x2,
            y2,
            confidence,
        )

        if is_plate_detection(
            detection
        ):

            candidates.append(
                detection
            )

    if not candidates:

        return None

    # Highest-confidence valid plate.
    candidates.sort(
        key=lambda x: x[4],
        reverse=True,
    )

    return candidates[0]


# ============================================================
# CROP
# ============================================================

def crop_plate(
    frame,
    detection,
):

    if detection is None:

        return None

    x1, y1, x2, y2, _ = detection

    h, w = frame.shape[:2]

    x1 = max(
        0,
        min(x1, w),
    )

    x2 = max(
        0,
        min(x2, w),
    )

    y1 = max(
        0,
        min(y1, h),
    )

    y2 = max(
        0,
        min(y2, h),
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):

        return None

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return None

    return crop


# ============================================================
# OCR
# ============================================================

def run_ocr(crop):

    if crop is None:

        return "", 0.0

    variants = []

    # 1. Original
    variants.append(
        ("original", crop)
    )

    # 2. Upscaled
    enlarged = cv2.resize(
        crop,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC,
    )

    variants.append(
        ("upscaled", enlarged)
    )

    # 3. Gray
    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    gray_bgr = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    variants.append(
        ("gray", gray_bgr)
    )

    # 4. Enhanced
    enhanced = cv2.detailEnhance(
        enlarged,
        sigma_s=10,
        sigma_r=0.15,
    )

    variants.append(
        ("enhanced", enhanced)
    )

    candidates = []

    for variant_name, image in variants:

        try:

            result = list(
                ocr(image)
            )

        except Exception as exc:

            print(
                f"OCR error "
                f"({variant_name}): {exc}"
            )

            continue

        if not result:

            continue

        for item in result:

            text = clean_text(
                item.get(
                    "rec_text",
                    "",
                )
            )

            try:

                confidence = float(
                    item.get(
                        "rec_score",
                        0.0,
                    )
                )

            except Exception:

                confidence = 0.0

            if text:

                candidates.append(
                    (
                        text,
                        confidence,
                        variant_name,
                    )
                )

    with shared_state.lock:

        shared_state.ocr_calls += len(
            variants
        )

    if candidates:

        print()

        print(
            "---------------- OCR DEBUG ----------------"
        )

        for (
            text,
            confidence,
            variant_name,
        ) in candidates:

            normalized = (
                normalize_plate_candidate(
                    text
                )
            )

            print(
                f"{variant_name:10s} "
                f"OCR={text:15s} "
                f"conf={confidence:.3f} "
                f"normalized={normalized}"
            )

        print(
            "-------------------------------------------"
        )

    if not candidates:

        return "", 0.0

    valid_candidates = []

    for (
        text,
        confidence,
        variant_name,
    ) in candidates:

        normalized = (
            normalize_plate_candidate(
                text
            )
        )

        if valid_kz_plate(
            normalized
        ):

            valid_candidates.append(
                (
                    normalized,
                    confidence,
                    variant_name,
                )
            )

    if valid_candidates:

        valid_candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        best = valid_candidates[0]

        print(
            f"BEST VALID OCR: "
            f"{best[0]} "
            f"conf={best[1]:.3f} "
            f"source={best[2]}"
        )

        return (
            best[0],
            best[1],
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    best = candidates[0]

    print(
        f"BEST RAW OCR: "
        f"{best[0]} "
        f"conf={best[1]:.3f} "
        f"source={best[2]}"
    )

    return (
        best[0],
        best[1],
    )


# ============================================================
# OCRM LOOKUP
# ============================================================

def lookup_ocrm(plate):

    plate = clean_text(plate)

    try:

        response = requests.get(
            f"{OCRM_URL}/api/vehicle/{plate}",
            timeout=3,
        )

    except requests.RequestException as exc:

        return (
            None,
            f"OCRM недоступен: {exc}",
        )

    if response.status_code == 200:

        try:

            payload = response.json()

        except Exception:

            return (
                None,
                "OCRM вернул некорректный JSON.",
            )

        return (
            payload.get("data"),
            "",
        )

    if response.status_code == 404:

        return (
            None,
            "NOT_FOUND",
        )

    return (
        None,
        f"OCRM HTTP error: "
        f"{response.status_code}",
    )


# ============================================================
# RESET
# ============================================================

def reset_for_new_vehicle():

    with shared_state.lock:

        shared_state.state = "SEARCH"

        shared_state.current_plate = ""

        shared_state.current_confidence = 0.0

        shared_state.last_ocr = ""

        shared_state.last_ocr_conf = 0.0

        shared_state.readings.clear()

        shared_state.lost_count = 0

        shared_state.last_detection = None

        shared_state.confirmed_detection = None

        shared_state.different_detection_count = 0

        shared_state.ocrm_data = None

        shared_state.ocrm_plate = ""

        shared_state.ocrm_error = ""

        shared_state.ocr_failure_message = ""


# ============================================================
# CONFIRM PLATE
# ============================================================

def confirm_plate(
    plate,
    confidence,
    detection,
):

    with shared_state.lock:

        if (
            shared_state.current_plate
            == plate
            and
            shared_state.state
            == "CONFIRMED"
        ):

            return

        shared_state.current_plate = plate

        shared_state.current_confidence = (
            confidence
        )

        shared_state.vehicle_count += 1

        shared_state.results.append(
            plate
        )

        shared_state.readings.clear()

        shared_state.state = "CONFIRMED"

        shared_state.lost_count = 0

        # IMPORTANT:
        # Remember bbox of the confirmed plate.
        shared_state.confirmed_detection = (
            detection
        )

        shared_state.last_detection = (
            detection
        )

        shared_state.different_detection_count = 0

        shared_state.ocr_failure_message = ""

        shared_state.ocrm_plate = ""

        shared_state.ocrm_data = None

        shared_state.ocrm_error = ""


# ============================================================
# RETURN TO SEARCH
# ============================================================

def return_to_search():

    print(
        ">>> Current plate lost. "
        "Returning to SEARCH."
    )

    with shared_state.lock:

        shared_state.state = "SEARCH"

        shared_state.current_plate = ""

        shared_state.current_confidence = 0.0

        shared_state.readings.clear()

        shared_state.last_ocr = ""

        shared_state.last_ocr_conf = 0.0

        shared_state.lost_count = 0

        shared_state.last_detection = None

        shared_state.confirmed_detection = None

        shared_state.different_detection_count = 0

        shared_state.ocrm_data = None

        shared_state.ocrm_plate = ""

        shared_state.ocrm_error = ""

        shared_state.ocr_failure_message = ""


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class LPRVideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        start_time = time.time()

        image = frame.to_ndarray(format="bgr24")

        with shared_state.lock:
            shared_state.frame_number += 1
            current_frame = shared_state.frame_number
            current_state = shared_state.state

        output = image.copy()

        # ====================================================
        # YOLO
        # ====================================================

        detection = None

        if current_frame % YOLO_FRAME_STEP == 0:

            raw_detection = detect_plate(image)

            with shared_state.lock:
                current_state = shared_state.state

                if current_state != "CONFIRMED":

                    # Stabilize detections before allowing OCR.
                    if raw_detection is None:

                        shared_state.detection_candidate = None
                        shared_state.detection_votes = 0
                        shared_state.last_detection = None
                        shared_state.lost_count += 1

                    else:

                        previous_candidate = (
                            shared_state.detection_candidate
                        )

                        if previous_candidate is None:

                            shared_state.detection_candidate = raw_detection
                            shared_state.detection_votes = 1
                            shared_state.lost_count = 0
                            shared_state.last_detection = None

                        elif detection_centers_close(
                            previous_candidate,
                            raw_detection,
                        ):

                            shared_state.detection_candidate = raw_detection
                            shared_state.detection_votes += 1
                            shared_state.lost_count = 0

                        else:

                            shared_state.detection_candidate = raw_detection
                            shared_state.detection_votes = 1
                            shared_state.lost_count = 0
                            shared_state.last_detection = None

                        if (
                            shared_state.detection_votes
                            >= DETECTION_VOTES
                        ):
                            shared_state.last_detection = (
                                shared_state.detection_candidate
                            )

                    detection = shared_state.last_detection

                else:

                    # CONFIRMED: track the confirmed plate.
                    confirmed_detection = (
                        shared_state.confirmed_detection
                    )

                    if raw_detection is None:

                        shared_state.lost_count += 1
                        shared_state.different_detection_count = 0

                    elif detections_belong_to_same_plate(
                        confirmed_detection,
                        raw_detection,
                    ):

                        shared_state.lost_count = 0
                        shared_state.different_detection_count = 0
                        shared_state.last_detection = raw_detection
                        shared_state.confirmed_detection = raw_detection

                    else:

                        shared_state.different_detection_count += 1
                        shared_state.lost_count = 0

                        if (
                            shared_state.different_detection_count
                            >= CONFIRMED_DIFFERENT_LIMIT
                        ):

                            # New object is present; forget old plate.
                            shared_state.state = "SEARCH"
                            shared_state.current_plate = ""
                            shared_state.current_confidence = 0.0
                            shared_state.readings.clear()
                            shared_state.last_ocr = ""
                            shared_state.last_ocr_conf = 0.0
                            shared_state.last_detection = None
                            shared_state.confirmed_detection = None
                            shared_state.different_detection_count = 0
                            shared_state.detection_candidate = raw_detection
                            shared_state.detection_votes = 1
                            shared_state.ocrm_data = None
                            shared_state.ocrm_plate = ""
                            shared_state.ocrm_error = ""
                            shared_state.ocr_failure_message = ""

                    detection = shared_state.last_detection

        else:

            with shared_state.lock:
                detection = shared_state.last_detection

        # ====================================================
        # VEHICLE LOST
        # ====================================================

        with shared_state.lock:
            lost_count = shared_state.lost_count
            state_now = shared_state.state

        if state_now == "CONFIRMED" and lost_count >= LOST_LIMIT:

            return_to_search()

            detection = None
            state_now = "SEARCH"

        # ====================================================
        # OCR DECISION
        # ====================================================

        should_ocr = False

        if detection is not None:

            if state_now == "SEARCH":

                # A stable detection is required before this point.
                # OCR can start immediately once it is available.
                should_ocr = True

            elif state_now == "RECOGNIZE":

                should_ocr = (
                    current_frame
                    % (YOLO_FRAME_STEP * OCR_FRAME_STEP)
                    == 0
                )

            # No OCR while CONFIRMED.

        # ====================================================
        # OCR
        # ====================================================

        if should_ocr:

            crop = crop_plate(image, detection)

            text, ocr_conf = run_ocr(crop)

            normalized = normalize_plate_candidate(text)

            with shared_state.lock:
                shared_state.last_ocr = text
                shared_state.last_ocr_conf = ocr_conf

                state_before_ocr = shared_state.state

            if state_before_ocr == "SEARCH":

                if (
                    valid_kz_plate(normalized)
                    and ocr_conf >= MIN_OCR_CONF
                ):

                    with shared_state.lock:
                        shared_state.readings.clear()
                        shared_state.readings.append(normalized)
                        shared_state.ocr_failure_message = ""
                        shared_state.state = "RECOGNIZE"

                else:

                    with shared_state.lock:
                        shared_state.ocr_failure_message = (
                            "ГРНЗ обнаружен, "
                            "ожидается распознавание номера..."
                        )

            elif state_before_ocr == "RECOGNIZE":

                if (
                    valid_kz_plate(normalized)
                    and ocr_conf >= MIN_OCR_CONF
                ):

                    with shared_state.lock:
                        shared_state.readings.append(normalized)
                        readings_copy = list(shared_state.readings)
                        shared_state.ocr_failure_message = ""

                    (
                        voted_plate,
                        vote_conf,
                        exact_votes,
                    ) = character_vote(readings_copy)

                    if (
                        valid_kz_plate(voted_plate)
                        and exact_votes >= CONFIRM_VOTES
                    ):

                        confirm_plate(
                            voted_plate,
                            vote_conf,
                            detection,
                        )

                else:

                    with shared_state.lock:
                        shared_state.ocr_failure_message = (
                            "ГРНЗ обнаружен, "
                            "но OCR пока не смог "
                            "уверенно прочитать номер."
                        )

        # ====================================================
        # DRAW YOLO BOX
        # ====================================================

        if detection is not None:

            x1, y1, x2, y2, yolo_conf = detection

            with shared_state.lock:
                display_state = shared_state.state
                display_plate = shared_state.current_plate
                display_confidence = shared_state.current_confidence

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            if display_state == "CONFIRMED":

                label = (
                    f"{display_plate} "
                    f"{display_confidence * 100:.0f}%"
                )

            elif display_state == "RECOGNIZE":

                label = "READING PLATE..."

            else:

                label = "PLATE DETECTED"

            cv2.putText(
                output,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ====================================================
        # STATUS OVERLAY
        # ====================================================

        with shared_state.lock:
            state_text = shared_state.state
            vehicle_count = shared_state.vehicle_count
            plate_text = shared_state.current_plate

        cv2.rectangle(
            output,
            (10, 10),
            (520, 110),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            f"STATE: {state_text}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"VEHICLES: {vehicle_count}",
            (20, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if plate_text:

            cv2.putText(
                output,
                f"PLATE: {plate_text}",
                (20, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

        # ====================================================
        # FPS
        # ====================================================

        elapsed = time.time() - start_time

        instant_fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0.0
        )

        with shared_state.lock:

            shared_state.processing_times.append(
                instant_fps
            )

            shared_state.fps = (
                sum(shared_state.processing_times)
                / len(shared_state.processing_times)
            )

        return frame.from_ndarray(
            output,
            format="bgr24"
        )


# ============================================================
# STREAMLIT UI REFRESH
# ============================================================

st_autorefresh(
    interval=1000,
    key="ocrm_refresh",
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "LPR settings"
)

st.sidebar.write(
    f"YOLO: every {YOLO_FRAME_STEP} frames"
)

st.sidebar.write(
    f"OCR: every {OCR_FRAME_STEP} YOLO cycle"
)

st.sidebar.write(
    f"Effective OCR interval: "
    f"{YOLO_FRAME_STEP * OCR_FRAME_STEP} frames"
)

st.sidebar.write(
    f"Minimum OCR confidence: "
    f"{MIN_OCR_CONF:.2f}"
)

st.sidebar.write(
    f"Confirmation votes: "
    f"{CONFIRM_VOTES}"
)

st.sidebar.write(
    f"Lost limit: "
    f"{LOST_LIMIT} YOLO cycles"
)

st.sidebar.divider()

st.sidebar.header(
    "OCRM"
)

st.sidebar.write(
    OCRM_URL
)


# ============================================================
# RESET BUTTON
# ============================================================

if st.sidebar.button(
    "🔄 Reset"
):

    reset_for_new_vehicle()

    with shared_state.lock:

        shared_state.frame_number = 0

        shared_state.yolo_calls = 0

        shared_state.ocr_calls = 0

        shared_state.vehicle_count = 0

        shared_state.results.clear()

        shared_state.processing_times.clear()

        shared_state.fps = 0.0

    st.rerun()


# ============================================================
# CAMERA
# ============================================================

st.subheader(
    "📹 Live camera"
)

st.info(
    "START+camera"
    "Для сложного номера держи его в кадре несколько секунд."
)


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


webrtc_streamer(
    key="kz-lpr-ocrm",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=LPRVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)


# ============================================================
# READ CURRENT STATE
# ============================================================

with shared_state.lock:

    current_state = (
        shared_state.state
    )

    current_plate = (
        shared_state.current_plate
    )

    current_confidence = (
        shared_state.current_confidence
    )

    vehicle_count = (
        shared_state.vehicle_count
    )

    ocrm_data = (
        shared_state.ocrm_data
    )

    ocrm_plate = (
        shared_state.ocrm_plate
    )

    ocrm_error = (
        shared_state.ocrm_error
    )

    ocr_failure_message = (
        shared_state.ocr_failure_message
    )

    last_ocr = (
        shared_state.last_ocr
    )

    last_ocr_conf = (
        shared_state.last_ocr_conf
    )

    yolo_calls = (
        shared_state.yolo_calls
    )

    ocr_calls = (
        shared_state.ocr_calls
    )

    fps_value = (
        shared_state.fps
    )

    results_copy = list(
        shared_state.results
    )


# ============================================================
# OCRM LOOKUP
# ============================================================

if (
    current_state == "CONFIRMED"
    and
    current_plate
    and
    ocrm_plate
    !=
    current_plate
):

    data, error = lookup_ocrm(
        current_plate
    )

    with shared_state.lock:

        shared_state.ocrm_data = data

        shared_state.ocrm_plate = (
            current_plate
        )

        shared_state.ocrm_error = (
            error
        )

    ocrm_data = data

    ocrm_error = error


# ============================================================
# STATUS
# ============================================================

st.divider()

st.subheader(
    "Current recognition"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "State",
        current_state
    )

with col2:

    st.metric(
        "License plate",
        current_plate
        if current_plate
        else "—"
    )

with col3:

    st.metric(
        "Confidence",
        f"{current_confidence:.1%}"
    )

with col4:

    st.metric(
        "Vehicles",
        vehicle_count
    )


# ============================================================
# OCR STATUS
# ============================================================

if (
    current_state == "RECOGNIZE"
    and
    ocr_failure_message
):

    st.warning(
        f"⚠️ {ocr_failure_message}"
    )


# ============================================================
# OCRM INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🏦 OCRM information"
)

if (
    current_state == "CONFIRMED"
    and
    current_plate
):

    if ocrm_error == "NOT_FOUND":

        st.error(
            "❌ Автомобиль не найден в базе OCRM"
        )

        st.warning(
            f"ГРНЗ **{current_plate}** "
            "успешно распознан, но такого "
            "автомобиля нет в локальной базе OCRM."
        )

    elif ocrm_error:

        st.warning(
            f"⚠️ {ocrm_error}"
        )

    elif ocrm_data is None:

        st.info(
            f"🔎 Ищу **{current_plate}** в OCRM..."
        )

    else:

        client = ocrm_data.get(
            "client",
            {}
        )

        loan = ocrm_data.get(
            "loan",
            {}
        )

        vehicle = ocrm_data.get(
            "vehicle",
            {}
        )

        pledge = ocrm_data.get(
            "pledge",
            {}
        )

        organization = ocrm_data.get(
            "organization",
            {}
        )

        left, right = (
            st.columns(2)
        )

        with left:

            st.markdown(
                "### 👤 Заёмщик"
            )

            st.write(
                f"**ФИО:** "
                f"{client.get('name', '—')}"
            )

            st.write(
                f"**ИИН:** "
                f"{client.get('iin', '—')}"
            )

            st.write(
                f"**Телефон:** "
                f"{client.get('phone', '—')}"
            )

            st.markdown(
                "### 💳 Кредит"
            )

            st.write(
                f"**ДБЗ:** "
                f"{loan.get('loan_id', '—')}"
            )

            st.write(
                f"**Статус:** "
                f"{loan.get('status', '—')}"
            )

            st.write(
                f"**Просрочка:** "
                f"{loan.get('overdue_days', 0)} дней"
            )

        with right:

            st.markdown(
                "### 🚗 Автомобиль"
            )

            st.write(
                f"**ГРНЗ:** "
                f"{ocrm_data.get('plate', '—')}"
            )

            st.write(
                f"**Марка:** "
                f"{vehicle.get('brand', '—')}"
            )

            st.write(
                f"**Модель:** "
                f"{vehicle.get('model', '—')}"
            )

            st.write(
                f"**Год:** "
                f"{vehicle.get('year', '—')}"
            )

            st.write(
                f"**Цвет:** "
                f"{vehicle.get('color', '—')}"
            )

            st.markdown(
                "### 🔐 Залог"
            )

            st.write(
                f"**Статус:** "
                f"{pledge.get('status', '—')}"
            )

            st.write(
                f"**ID залога:** "
                f"{pledge.get('collateral_id', '—')}"
            )

            st.write(
                f"**Филиал:** "
                f"{organization.get('branch', '—')}"
            )

            st.write(
                f"**Ответственный:** "
                f"{organization.get('manager', '—')}"
            )

        st.success(
            f"✓ {current_plate} найден в OCRM"
        )

        # ====================================================
        # VISIT
        # ====================================================

        st.divider()

        st.subheader(
            "📋«Выезд kettik»"
        )

        visit_result = st.selectbox(
            "Rez выезда",
            [
                "Автомобиль обнаружен",
                "Контакт с клиентом установлен",
                "Контакт с клиентом не установлен",
                "Автомобиль не обнаружен",
            ],
        )

        visit_comment = st.text_area(
            "Комментарий",
            placeholder=(
                "Введите комментарий сотрудника..."
            ),
        )

        if st.button(
            "➕  «Выезд»",
            type="primary",
        ):

            try:

                response = requests.post(
                    f"{OCRM_URL}/api/visit",
                    json={
                        "plate": current_plate,
                        "result": visit_result,
                        "comment": visit_comment,
                    },
                    timeout=3,
                )

                if response.status_code == 200:

                    st.success(
                        "✓ «Выезд» создано"
                    )

                    st.json(
                        response.json()
                    )

                else:

                    st.error(
                        f"OCRM error: "
                        f"{response.status_code}"
                    )

            except requests.RequestException as exc:

                st.error(
                    f"OCRM недоступен: {exc}"
                )


# ============================================================
# HISTORY
# ============================================================

st.divider()

st.subheader(
    "🚗 Распознанные автомобили"
)

if results_copy:

    for index, plate in enumerate(
        results_copy,
        1
    ):

        st.success(
            f"{index}. {plate}"
        )

else:

    st.caption(
        "Пока нет подтверждённых автомобилей."
    )


# ============================================================
# TECHNICAL INFORMATION
# ============================================================

with st.expander(
    "🔧 Technical information"
):

    st.write(
        f"State: `{current_state}`"
    )

    st.write(
        f"Last OCR: `{last_ocr}`"
    )

    st.write(
        f"OCR confidence: "
        f"{last_ocr_conf:.3f}"
    )

    st.write(
        f"YOLO calls: {yolo_calls}"
    )

    st.write(
        f"OCR calls: {ocr_calls}"
    )

    st.write(
        f"Processing FPS: {fps_value:.1f}"
    )

    st.write(
        f"OCRM URL: `{OCRM_URL}`"
    )