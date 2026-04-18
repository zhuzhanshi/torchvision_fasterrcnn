from __future__ import annotations

from PIL import Image, ImageDraw


def draw_predictions(image: Image.Image, boxes, labels, scores, class_names, score_fmt="{:.2f}"):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        cls_name = class_names[label - 1] if 0 < label <= len(class_names) else str(label)
        text = f"{cls_name}:{score_fmt.format(float(score))}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, y1), text, fill="yellow")
    return img
