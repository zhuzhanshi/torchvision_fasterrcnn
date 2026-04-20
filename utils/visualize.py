from __future__ import annotations

from PIL import Image, ImageDraw


def draw_predictions(
    image: Image.Image,
    boxes,
    labels,
    scores,
    class_names,
    draw_label: bool = True,
    draw_score: bool = True,
    line_thickness: int = 2,
    score_fmt: str = "{:.2f}",
):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(boxes, labels, scores):
        if int(label) <= 0:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        cls_name = class_names[label - 1] if 0 < label <= len(class_names) else str(label)
        texts = []
        if draw_label:
            texts.append(str(cls_name))
        if draw_score:
            texts.append(score_fmt.format(float(score)))
        text = ":".join(texts)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=max(1, int(line_thickness)))
        if text:
            draw.text((x1, y1), text, fill="yellow")
    return img
