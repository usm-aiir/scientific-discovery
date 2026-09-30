from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

IMAGE_PATH = "/home/adah.holt/scientific-discovery/2504.00002_7.png"
TEXT_PROMPT = "legend . axis label . data point . trend line . title . scatter plot . x axis . y axis"

model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "grounding_dino_weights/groundingdino_swint_ogc.pth"
)

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=0.20,
    text_threshold=0.15
)

print("Detected:", phrases)
print("Confidence scores:", logits)

annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("/home/adah.holt/scientific-discovery/test_annotated.png", annotated)
print("Saved annotated image to test_annotated.png")