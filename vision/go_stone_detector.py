import os
import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv
from supervision.draw.color import ColorPalette

# ===== CONFIG =====
IMG_PATH = "images/image0.jpg"
BOARD_PROMPT = "Big Rectangular gridded Box."
STONE_PROMPT = "circular white stone."
OUTPUT_DIR = "images"

SAM2_CHECKPOINT = "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
# ==================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Build models
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

image = Image.open(IMG_PATH)
img_np = np.array(image.convert("RGB"))


def detect(prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    return processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]


def segment(boxes):
    sam2_predictor.set_image(img_np)
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    return masks


# Step 1: Detect and segment the board to create exclusion mask
print("Step 1: Detecting board region...")
board_results = detect(BOARD_PROMPT)
board_boxes = board_results["boxes"].cpu().numpy()
print(f"  Found {len(board_boxes)} board region(s)")

if len(board_boxes) == 0:
    print("  No board detected, skipping masking step.")
    board_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
else:
    board_masks = segment(board_boxes)
    # Combine all board masks into one exclusion zone
    board_mask = np.any(board_masks.astype(bool), axis=0)
    print(f"  Board mask covers {board_mask.sum()} pixels")

# Step 2: Detect stones
print("Step 2: Detecting stones...")
stone_results = detect(STONE_PROMPT)
stone_boxes = stone_results["boxes"].cpu().numpy()
stone_confidences = stone_results["scores"].cpu().numpy().tolist()
stone_labels = stone_results["labels"]
print(f"  Found {len(stone_boxes)} stone(s) total")

if len(stone_boxes) == 0:
    print("No stones detected. Try adjusting STONE_PROMPT or thresholds.")
    exit()

# Step 3: Segment stones and filter out those inside the board
stone_masks = segment(stone_boxes)

keep = []
for i, mask in enumerate(stone_masks):
    mask_bool = mask.astype(bool)
    overlap = np.logical_and(mask_bool, board_mask).sum()
    total = mask_bool.sum()
    overlap_ratio = overlap / total if total > 0 else 0
    # Keep stones that are mostly outside the board
    if overlap_ratio < 0.5:
        keep.append(i)
        print()
    else:
        print()

if len(keep) == 0:
    print("All stones were inside the board region. None remaining.")
    exit()

# Filter to only kept stones
final_boxes = stone_boxes[keep]
final_masks = stone_masks[keep]
final_confidences = [stone_confidences[i] for i in keep]
final_labels = [f"{stone_labels[i]} {stone_confidences[i]:.2f}" for i in keep]
final_class_ids = np.arange(len(keep))

# Visualize
img = cv2.imread(IMG_PATH)
detections = sv.Detections(
    xyxy=final_boxes,
    mask=final_masks.astype(bool),
    class_id=final_class_ids,
)

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
color_palette = ColorPalette.from_hex(COLORS)

annotated = img.copy()

# Add light board mask overlay
if board_mask.any():
    overlay = annotated.copy()
    board_color = (200, 230, 255)  # Light blue (BGR)
    overlay[board_mask] = board_color
    alpha = 0.25  # Light transparency
    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)
    # Add "board" label at board center
    ys, xs = np.where(board_mask)
    if len(ys) > 0:
        cy, cx = int(ys.mean()), int(xs.mean())
        cv2.putText(
            annotated, "board", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 150, 200), 2
        )

annotated = sv.BoxAnnotator(color=color_palette).annotate(scene=annotated, detections=detections)
annotated = sv.LabelAnnotator(color=color_palette).annotate(
    scene=annotated, detections=detections, labels=final_labels
)
annotated = sv.MaskAnnotator(color=color_palette).annotate(scene=annotated, detections=detections)

img_base = os.path.splitext(os.path.basename(IMG_PATH))[0]
output_path = os.path.join(OUTPUT_DIR, f"{img_base}_grounded.jpg")
cv2.imwrite(output_path, annotated)
print(f"\nSaved to {output_path} ({len(keep)} stones outside board)")
