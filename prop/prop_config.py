"""Propagation thresholds and constants. Module-level so propagation.py can import freely.

Override at runtime by assigning new values to attributes of this module
(see inference_gmos.py)."""

# ── Thresholds ────────────────────────────────────────────────────────────────
MOTION_THRES = 0.5          # minimum motion score to consider an object moving
IOU_THRES = 0.7             # minimum predicted IoU to trust a mask
MATCH_IOU_THRES = 0.95      # IoU between pred and SAM2 mask to reinforce existing object
MIN_PRECISION_THRES = 0.3   # max precision below which a new object is added
MOTION_PRECISION_THRES = 0.5  # precision threshold for motion label transfer
MOTION_IOU_THRES = 0.2      # final motion gating threshold
MIN_MOV_OBJECT_FRAME_PROP = 0.03  # min proportion of frames an object must be moving to keep in full_segments
MAX_OBJECTS = 10             # maximum number of tracked objects

# ── Temporal smoothing weights ────────────────────────────────────────────────
# 5-frame window with offsets [-2, -1, 0, +1, +2]
STAGE1_SMOOTHING_WEIGHTS = [0, 0, 1.0, 0.7, 0.4]       # asymmetric (causal-ish)
STAGE2_SMOOTHING_WEIGHTS = [0.25, 0.5, 1.0, 0.5, 0.25]  # symmetric

# ── Prompt re-injection ──────────────────────────────────────────────────────
MIN_PROMPT_FRAME_DISTANCE = 5  # minimum frames between prompts for same object
STAGE2_PROMPT_FRACTION = 0.1   # fraction of frames to use as prompts in stage 2
STAGE2_PROMPT_IOU_THRES = 0.95 # minimum IoU to qualify as stage 2 prompt
