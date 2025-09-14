import torch
import os
from PIL import Image
from ultralytics import YOLO
import numpy as np


CONFIDENCE_THRESHOLD = 0.3  # минимальная уверенность YOLO для всех классов
MIN_AREA_RATIO_DEFAULT = 0.005
MIN_AREA_RATIO_SCRATCH = 0.001
MAX_AREA_RATIO = 0.4
IGNORE_DAMAGE_PERCENT = 1.0
AVG_CONFIDENCE_MIN = 0.5

BOX_CONFIDENCE_MIN_DEFAULT = 0.2
BOX_CONFIDENCE_MIN_SCRATCH = 0.2

CLASSES = {
    0: "scratch",
    1: "dent",
    2: "crack",
    3: "rust",
    4: "broken_part"
}

base_dir = os.path.dirname(__file__)
weights_path = os.path.join(base_dir, "models/yolov8n.pt")

model = None

def convert_to_json_compatible(obj):
    """Конвертирует numpy типы и другие объекты в JSON-совместимые типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_compatible(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_compatible(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_compatible(item) for item in obj)
    else:
        return obj

# =========================
# Загрузка модели
# =========================
def load_model():
    global model
    if model is None:
        print("🔄 Загружаем YOLO модель...")
        if not os.path.exists(weights_path):
            print("⚠️ Обученная модель не найдена, используем предобученную YOLOv8")
            model = YOLO('yolov8n.pt')
        else:
            try:
                model = YOLO(weights_path)
                print(f"✅ Загружена обученная модель: {weights_path}")
            except Exception as e:
                print(f"❌ Ошибка загрузки обученной модели: {e}")
                model = YOLO('yolov8n.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print(f"🚀 Модель загружена на устройство: {device}")
    return model


def predict_damage(img_path):
    yolo_model = load_model()
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Файл изображения не найден: {img_path}")

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    results = yolo_model.predict(
        source=img_path,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.3,
        verbose=False,
        save=False,
        show=False
    )

    detections = []

    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                class_id = int(cls)
                damage_type = CLASSES.get(class_id, f"unknown_class_{class_id}")

                min_area = MIN_AREA_RATIO_SCRATCH * img_w * img_h if damage_type == "scratch" else MIN_AREA_RATIO_DEFAULT * img_w * img_h
                box_conf_min = BOX_CONFIDENCE_MIN_SCRATCH if damage_type == "scratch" else BOX_CONFIDENCE_MIN_DEFAULT

                if float(conf) < box_conf_min:
                    continue

                x1, y1, x2, y2 = map(float, box)
                if x2 <= x1 or y2 <= y1:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if not (min_area <= area <= MAX_AREA_RATIO * img_w * img_h):
                    continue

                detections.append({
                    "type": damage_type,
                    "confidence": round(float(conf) * 100, 2),
                    "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "area": round(float(area), 2)
                })

    return detections


def predict_damage_with_metadata(img_path):
    detections = predict_damage(img_path)
    img = Image.open(img_path)
    img_w, img_h = img.size

    damage_stats = {}
    total_damage_area = 0.0
    confidences_list = []

    for d in detections:
        damage_type = str(d["type"])
        damage_stats[damage_type] = damage_stats.get(damage_type, 0) + 1
        total_damage_area += float(d["area"])
        confidences_list.append(float(d["confidence"]) / 100.0)

    image_area = float(img_w * img_h)
    damage_percentage = (total_damage_area / image_area) * 100 if image_area > 0 else 0.0
    avg_confidence = float(np.mean(confidences_list)) if confidences_list else 0.0

    if len(detections) == 0 or damage_percentage < IGNORE_DAMAGE_PERCENT or avg_confidence < AVG_CONFIDENCE_MIN:
        severity = "none"
    elif damage_percentage <= 5:
        severity = "low"
    elif damage_percentage <= 15:
        severity = "medium"
    else:
        severity = "high"

    result = {
        "image_info": {
            "path": str(img_path), 
            "width": int(img_w), 
            "height": int(img_h), 
            "area": int(image_area)
        },
        "damage_summary": {
            "total_detections": int(len(detections)),
            "damage_types": damage_stats,
            "total_damage_area": round(float(total_damage_area), 2),
            "damage_percentage": round(float(damage_percentage), 2),
            "average_confidence": round(float(avg_confidence * 100), 2),
            "severity": str(severity)
        },
        "detections": detections
    }
    
    return convert_to_json_compatible(result)


if __name__ == "__main__":
    pass