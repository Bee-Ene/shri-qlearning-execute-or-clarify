"""
shri_state.py — State Abstraction and NLU Parser — FINAL VERSION
Shared between Colab simulation and ROS2 deployment.

Key fix: parse_confidence for complete commands is 0.92 (CLEAR).
Only missing params reduce confidence below 0.60 threshold.
"move forward" → direction present → conf=0.92 → CLEAR → State 0.
"""

from enum import Enum
from typing import Optional, Tuple
import re


# ── State dimensions ──────────────────────────────────────────────────────────

class Completeness(Enum):
    COMPLETE   = 0
    PARTIAL    = 1
    INCOMPLETE = 2

class Clarity(Enum):
    CLEAR     = 0
    AMBIGUOUS = 1

class Feasibility(Enum):
    FEASIBLE  = 0
    UNCERTAIN = 1
    BLOCKED   = 2

# Thresholds — confirmed stable by sensitivity analysis (std=0.020)
PARSE_AMBIGUOUS_THRESHOLD  = 0.60
PERC_UNCERTAIN_THRESHOLD   = 0.45
PERC_UNCERTAIN_NAV         = 0.50
BLOCKED_DISTANCE_THRESHOLD = 0.70   # metres

UNAMBIGUOUS_INTENTS = {"STOP", "SIT", "STAND"}

REQUIRED_PARAMS = {
    "MOVE":        ["direction"],
    "TURN":        ["direction"],
    "NAVIGATE_TO": ["target"],
    "STOP":        [],
    "SIT":         [],
    "STAND":       [],
}

NUM_STATES  = 18
NUM_ACTIONS = 2

# Direction aliases
DIRECTIONS = [
    "forward", "backwards", "backward", "back",
    "left", "right", "ahead", "straight"
]
DIRECTION_MAP = {
    "backwards": "backward",
    "back":      "backward",
    "ahead":     "forward",
    "straight":  "forward",
}

# TTS echo detection — robot should not command itself
TTS_ECHO_PHRASES = [
    "ready", "please give", "understood", "executing",
    "which direction", "where should", "obstacle",
    "task complete", "episode", "moving to clear",
    "clarify", "rephrase", "i did not", "rotating",
    "moving forward", "moving backward",
]


# ── State encoding / decoding ─────────────────────────────────────────────────

def encode_state(c: Completeness, cl: Clarity, f: Feasibility) -> int:
    return c.value * 6 + cl.value * 3 + f.value

def decode_state(s: int) -> Tuple[Completeness, Clarity, Feasibility]:
    return (Completeness(s // 6),
            Clarity((s % 6) // 3),
            Feasibility(s % 3))

def state_to_string(state_id: int) -> str:
    c, cl, f = decode_state(state_id)
    return f"{c.name}+{cl.name}+{f.name}"

def state_uncertainty_level(s: int) -> int:
    """Scalar 0-5. Higher = more uncertain. Used for outcome-based reward."""
    c, cl, f = decode_state(s)
    score = 0
    if c  == Completeness.INCOMPLETE: score += 2
    elif c == Completeness.PARTIAL:   score += 1
    if cl == Clarity.AMBIGUOUS:       score += 1
    if f  == Feasibility.BLOCKED:     score += 2
    elif f == Feasibility.UNCERTAIN:  score += 1
    return score


# ── State builder ─────────────────────────────────────────────────────────────

def build_state(intent: str,
                present_params: list,
                missing_params: list,
                parse_confidence: float,
                path_clear: bool,
                target_visible: bool,
                num_target_matches: int,
                perception_confidence: float) -> int:

    required = REQUIRED_PARAMS.get(intent, [])

    # Completeness: from argument presence only
    if not required:
        comp = Completeness.COMPLETE
    else:
        n_pres = len(present_params)
        n_req  = len(required)
        if   n_pres == n_req: comp = Completeness.COMPLETE
        elif n_pres == 0:     comp = Completeness.INCOMPLETE
        else:                 comp = Completeness.PARTIAL

    # Clarity: from parse confidence only
    if intent in UNAMBIGUOUS_INTENTS:
        clar = Clarity.CLEAR
    else:
        clar = (Clarity.AMBIGUOUS
                if parse_confidence < PARSE_AMBIGUOUS_THRESHOLD
                else Clarity.CLEAR)

    # Feasibility: from sensor data only
    needs_target = (intent == "NAVIGATE_TO")
    if not path_clear:
        feas = Feasibility.BLOCKED
    elif needs_target:
        if not target_visible:
            feas = Feasibility.BLOCKED
        elif num_target_matches > 1:
            feas = Feasibility.UNCERTAIN
        elif perception_confidence < PERC_UNCERTAIN_NAV:
            feas = Feasibility.UNCERTAIN
        else:
            feas = Feasibility.FEASIBLE
    else:
        feas = (Feasibility.UNCERTAIN
                if perception_confidence < PERC_UNCERTAIN_THRESHOLD
                else Feasibility.FEASIBLE)

    return encode_state(comp, clar, feas)


# ── NLU Parser — FINAL ────────────────────────────────────────────────────────

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "half": 0.5,
}

TARGETS = ["chair", "door", "table", "person", "marker"]


def _preprocess(text: str) -> str:
    """Lowercase, convert number words to digits."""
    text = text.strip().lower()
    words = text.split()
    out = []
    for w in words:
        out.append(str(WORD_TO_NUM[w]) if w in WORD_TO_NUM else w)
    return " ".join(out)


def is_tts_echo(text: str) -> bool:
    """Returns True if text looks like robot's own TTS output."""
    t = text.lower()
    return any(phrase in t for phrase in TTS_ECHO_PHRASES)


def parse_command(text: str) -> dict:
    """
    Rule-based NLU for locomotive commands. FINAL VERSION.

    Confidence rules (critical — determines CLEAR vs AMBIGUOUS):
      - All required args present, unambiguous intent → 0.92 (CLEAR)
      - One required arg missing                      → 0.55 (AMBIGUOUS)
      - Two required args missing                     → 0.40 (AMBIGUOUS)
      - STOP/SIT/STAND (no args needed)               → 0.95 (CLEAR)
      - Unknown intent                                → 0.10

    "move forward" → direction=forward, missing=[], conf=0.92 → CLEAR → state 0
    "move"         → direction=None,    missing=["direction"], conf=0.55 → AMBIGUOUS
    "go to chair"  → NAVIGATE_TO, target=chair, missing=[], conf=0.92 → CLEAR
    "go left"      → MOVE, direction=left, missing=[], conf=0.92 → CLEAR
    """
    text = _preprocess(text)

    # ── Detect direction ──────────────────────────────────────────────
    direction = None
    words = text.split()
    for d in DIRECTIONS:
        if d in words:
            direction = DIRECTION_MAP.get(d, d)
            break

    # ── Detect target ──────────────────────────────────────────────────
    target = None
    for t in TARGETS:
        if t in text:
            target = "the " + t
            break

    # ── Detect distance ───────────────────────────────────────────────
    distance = None
    m = re.search(r"(\d+\.?\d*)\s*(metre|meter|m\b)", text)
    if m:
        distance = float(m.group(1))

    # ── Detect intent ─────────────────────────────────────────────────
    # NAVIGATE_TO: only when explicit "go to" + target OR "navigate to"
    # "go left/right/forward/backward" → MOVE, NOT NAVIGATE_TO
    intent = "UNKNOWN"

    nav_phrases = ["go to", "navigate to", "take me to", "walk to"]
    is_nav = any(ph in text for ph in nav_phrases) and target is not None

    if is_nav:
        intent    = "NAVIGATE_TO"
        direction = None

    elif any(w in words for w in
             ["move", "walk", "go", "travel", "proceed", "advance"]):
        intent = "MOVE"

    elif any(w in words for w in
             ["turn", "rotate", "spin", "face"]):
        intent = "TURN"

    elif any(w in words for w in
             ["stop", "halt", "freeze", "wait", "stay"]):
        intent = "STOP"

    elif any(w in words for w in
             ["sit", "crouch", "down"]):
        intent = "SIT"

    elif any(w in words for w in
             ["stand", "rise", "up", "get up"]):
        intent = "STAND"

    # ── Argument tracking ─────────────────────────────────────────────
    required       = REQUIRED_PARAMS.get(intent, [])
    present_params = []
    missing_params = []

    if "direction" in required:
        if direction is not None:
            present_params.append("direction")
        else:
            missing_params.append("direction")

    if "target" in required:
        if target is not None:
            present_params.append("target")
        else:
            missing_params.append("target")

    # ── Confidence — determines CLEAR vs AMBIGUOUS ────────────────────
    # This is the critical computation. Must be >= 0.60 for CLEAR.
    n_miss = len(missing_params)

    if intent == "UNKNOWN":
        conf = 0.10
    elif intent in UNAMBIGUOUS_INTENTS:
        conf = 0.95   # STOP/SIT/STAND always CLEAR
    elif n_miss == 0:
        conf = 0.92   # all args present → CLEAR
    elif n_miss == 1:
        conf = 0.55   # one arg missing → AMBIGUOUS (needs clarification)
    else:
        conf = 0.40   # multiple args missing → AMBIGUOUS

    return {
        "intent":           intent,
        "direction":        direction,
        "target":           target,
        "distance":         distance,
        "present_params":   present_params,
        "missing_params":   missing_params,
        "parse_confidence": conf,
        "raw_text":         text,
    }


def generate_clarification_question(intent: str,
                                     missing_params: list,
                                     feasibility: Feasibility) -> str:
    """Generates a natural, targeted clarification question."""
    if feasibility == Feasibility.BLOCKED:
        return ("There is an obstacle in my path. "
                "Which direction should I go to avoid it? "
                "Say backward, left, or right.")
    if "direction" in missing_params:
        if intent == "MOVE":
            return ("Which direction should I move? "
                    "Say forward, backward, left, or right.")
        elif intent == "TURN":
            return "Which way should I turn? Say left or right."
    if "target" in missing_params:
        return ("Where should I go? "
                "Say the chair, the door, the table, or the marker.")
    return ("Could you clarify your command? "
            "Try saying: move forward, turn left, or go to the chair.")
