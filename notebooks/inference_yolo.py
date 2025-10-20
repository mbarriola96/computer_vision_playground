# ────────────────────────────────────────────────
# 🔍 Inference Script for NEU-DET YOLOv8 model
# ────────────────────────────────────────────────

from ultralytics import YOLO
import os

# 1️⃣ Ruta al modelo entrenado
model_path = r"runs/detect/NEUDET_yolov8n9/weights/best.pt"

# 2️⃣ Ruta a la imagen sobre la que quieres hacer inferencia
# 👉 Cambia este path por tu imagen real
image_path = r"../data/NEU-DET/validation/images/patches/patches_241.jpg"

# 3️⃣ Crear carpeta de salida (opcional)
output_dir = r"runs/inference_results"
os.makedirs(output_dir, exist_ok=True)

# 4️⃣ Cargar el modelo
model = YOLO(model_path)

# 5️⃣ Hacer predicción
results = model.predict(
    source=image_path,   # puede ser también un folder o '*.jpg'
    conf=0.25,           # nivel de confianza mínimo
    save=True,           # guarda la imagen con bounding boxes
    save_txt=False,      # guarda solo imagen, no texto
    project=output_dir,  # dónde guardar resultados
    name="NEUDET_inference",  # subcarpeta de resultados
    show=False           # cambia a True si quieres mostrar la imagen
)

# 6️⃣ Mostrar resultados en consola
for r in results:
    boxes = r.boxes.xyxy  # coordenadas [x1, y1, x2, y2]
    cls = r.boxes.cls     # clases detectadas
    conf = r.boxes.conf   # confianza
    print("\n🧩 Detecciones:")
    for i in range(len(boxes)):
        print(f"Clase: {model.names[int(cls[i])]} | Confianza: {float(conf[i]):.2f} | BBox: {boxes[i].tolist()}")

print("\n✅ Resultado guardado en:", results[0].save_dir)
