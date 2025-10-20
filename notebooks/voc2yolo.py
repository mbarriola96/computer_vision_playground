#!/usr/bin/env python3
"""
Convert NEU-DET (VOC XML) annotations to YOLO format.

- Lee todos los .xml en --ann_dir
- Escribe un .txt por imagen en --out_labels (por defecto: <img_dir>/../labels)
- Usa clases de NEU-DET por defecto, pero admite clases nuevas si aparecen.
- Si el <size> no está en el XML, intenta abrir la imagen para obtener (w,h).

Uso:
python voc2yolo_neudet.py \
  --ann_dir /ruta/NEU-DET/annotations \
  --img_dir /ruta/NEU-DET/images \
  --out_labels /ruta/NEU-DET/labels

Opcional: genera un data.yaml con --make_yaml y --dataset_root.
"""

import argparse
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # Solo necesario si falta <size> en el XML

# Clases oficiales de NEU-DET (orden fijo):
NEUDET_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True, help="Carpeta con XML (VOC).")
    ap.add_argument("--img_dir", required=True, help="Carpeta con imágenes.")
    ap.add_argument("--out_labels", default=None,
                    help="Carpeta destino para .txt (por defecto: <img_dir>/../labels)")
    ap.add_argument("--dataset_root", default=None,
                    help="Ruta raíz del dataset para data.yaml (si usas --make_yaml).")
    ap.add_argument("--make_yaml", action="store_true",
                    help="Crear data.yaml (requiere --dataset_root).")
    ap.add_argument("--names_file", default=None,
                    help="Guardar names.txt con las clases detectadas.")
    return ap.parse_args()

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def get_image_size_from_xml_or_file(xml_root, img_path):
    # Intenta del XML:
    size = xml_root.find("size")
    if size is not None:
        w = int(size.findtext("width", default="0"))
        h = int(size.findtext("height", default="0"))
        if w > 0 and h > 0:
            return w, h
    # Si falta o es 0, intenta abrir la imagen:
    if Image is None:
        raise RuntimeError("PIL no instalado y el XML no trae <size>; instala pillow o corrige el XML.")
    with Image.open(img_path) as im:
        return im.width, im.height

def clip(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    # Asegura orden y recorte a límites
    xmin, xmax = sorted([xmin, xmax])
    ymin, ymax = sorted([ymin, ymax])
    xmin = clip(xmin, 0, img_w - 1)
    xmax = clip(xmax, 0, img_w - 1)
    ymin = clip(ymin, 0, img_h - 1)
    ymax = clip(ymax, 0, img_h - 1)

    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        return None  # caja inválida

    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0

    # Normaliza a [0,1]
    return (cx / img_w, cy / img_h, bw / img_w, bh / img_h)

def load_class_map():
    # Mapa inicial con clases NEU-DET
    return {name: idx for idx, name in enumerate(NEUDET_CLASSES)}

def main():
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.img_dir)
    out_labels = Path(args.out_labels) if args.out_labels else (img_dir.parent / "labels")

    ensure_dir(out_labels)

    class_map = load_class_map()
    seen_classes = set()

    xml_files = sorted(glob.glob(str(ann_dir / "*.xml")))
    if not xml_files:
        print(f"[WARN] No se encontraron XML en {ann_dir}")
        return

    num_xml = len(xml_files)
    num_boxes = 0
    skipped = 0

    for i, xml_path in enumerate(xml_files, 1):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename")
        if not filename:
            # Algunos VOC usan <path> o <filename> distinto; intenta derivar del XML
            filename = Path(xml_path).stem + ".jpg"

        # Busca recursivamente la imagen en subcarpetas si no está directamente
        img_path = None
        for p in img_dir.rglob(filename):
            img_path = p
            break
        if img_path is None:
            print(f"[WARN] Imagen no encontrada para {xml_path} -> {filename}")
            skipped += 1
            continue

        if not img_path.exists():
            # Prueba extensiones comunes
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
                alt = img_dir / (Path(filename).stem + ext)
                if alt.exists():
                    img_path = alt
                    break
        if not img_path.exists():
            print(f"[WARN] Imagen no encontrada para {xml_path} -> {filename}")
            skipped += 1
            continue

        img_w, img_h = get_image_size_from_xml_or_file(root, img_path)

        # archivo de salida .txt con el mismo nombre base que la imagen
        out_txt = out_labels / (img_path.stem + ".txt")

        lines = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name is None:
                continue
            seen_classes.add(name)
            if name not in class_map:
                # Si aparece una clase nueva no esperada, la añadimos al final
                class_map[name] = len(class_map)

            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = float(bbox.findtext("xmin", default="0"))
            ymin = float(bbox.findtext("ymin", default="0"))
            xmax = float(bbox.findtext("xmax", default="0"))
            ymax = float(bbox.findtext("ymax", default="0"))

            yolo = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            if yolo is None:
                continue

            cls_id = class_map[name]
            cx, cy, bw, bh = yolo
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            num_boxes += 1

        # Escribe (vacío si no hay objetos -> archivo 0-bytes, YOLO lo admite)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if i % 100 == 0 or i == num_xml:
            print(f"[{i}/{num_xml}] {Path(xml_path).name} -> {out_txt.name}")

    # Guardar names.txt si se pidió
    if args.names_file:
        with open(args.names_file, "w", encoding="utf-8") as f:
            names_sorted = [None] * len(class_map)
            # invertimos el mapa para escribir en orden por id
            for k, v in class_map.items():
                if v < len(names_sorted):
                    names_sorted[v] = k
                else:
                    names_sorted.append(k)
            # Rellena posibles None si hubo ids saltados (no debería)
            names_sorted = [n if n is not None else f"class_{i}" for i, n in enumerate(names_sorted)]
            f.write("\n".join(names_sorted))
        print(f"[INFO] Guardado names.txt en {args.names_file}")

    # data.yaml opcional
    if args.make_yaml:
        if not args.dataset_root:
            print("[WARN] --make_yaml requiere --dataset_root")
        else:
            yaml_path = Path(args.dataset_root) / "data.yaml"
            # Ordena por id
            names_by_id = [None] * len(class_map)
            for name, idx in class_map.items():
                if idx < len(names_by_id):
                    names_by_id[idx] = name
                else:
                    names_by_id.append(name)
            names_by_id = [n if n is not None else f"class_{i}" for i, n in enumerate(names_by_id)]
            content = (
                f"path: {Path(args.dataset_root).as_posix()}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"names: {names_by_id}\n"
            )
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[INFO] data.yaml creado en {yaml_path}")

    print(f"\n[RESUMEN] XML procesados: {num_xml} | cajas válidas: {num_boxes} | XML sin imagen: {skipped}")
    print(f"[CLASES DETECTADAS] {sorted(list(seen_classes))}")
    print(f"[MAPA CLASES] " + ", ".join(f"{k}:{v}" for k, v in sorted(class_map.items(), key=lambda x: x[1])))

if __name__ == "__main__":
    main()
