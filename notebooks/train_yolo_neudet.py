import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# 1️⃣ Carga el modelo preentrenado (por ejemplo, YOLOv8n)
model = YOLO("yolov8n.pt")

# 2️⃣ Entrenamiento
model.train(
    data="neu-det.yaml",     # archivo YAML
    epochs=50,               # número de épocas
    imgsz=640,               # tamaño de las imágenes
    batch=16,                # tamaño del batch
    name="NEUDET_yolov8n",   # nombre del experimento
    workers=4,
    patience=10              # early stopping
)

# 3️⃣ Validación posterior
metrics = model.val()
print(metrics)
