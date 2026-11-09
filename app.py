from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Lock

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "YOLO11n": ROOT / "models" / "yolo11n_best.pt",
    "YOLOv5s": ROOT / "models" / "yolov5s_best.pt",
}
YOLOV5_REPO = ROOT / "yolov5"
CLASSES = {"ciherang", "ir64", "mentik"}
COLORS = {"ciherang": "#26B9E8", "ir64": "#46D899", "mentik": "#F6BB4B"}
IMAGE_SIZE = 800
MAX_DETECTIONS = 1000
MAX_PIXELS = 24_000_000


@st.cache_resource(show_spinner=False)
def load_model(model_name, checkpoint_path, checkpoint_stamp):
    """Cache by checkpoint path and version so replacing a weight reloads it."""
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if model_name == "YOLOv5s":
        if not (YOLOV5_REPO / "hubconf.py").is_file():
            raise FileNotFoundError("Folder yolov5 belum tersedia. Ikuti langkah YOLOv5s di README.")
        model = torch.hub.load(
            str(YOLOV5_REPO), "custom", path=checkpoint_path, source="local", device=device
        )
        model.eval()
    else:
        from ultralytics import YOLO

        model = YOLO(checkpoint_path)
        if model.task != "detect":
            raise ValueError("Checkpoint harus berasal dari model deteksi objek.")

    names = model.names
    class_names = names.values() if isinstance(names, dict) else names
    if {str(name).strip().lower() for name in class_names} != CLASSES:
        raise ValueError("Gunakan best.pt deteksi Ciherang, IR64, dan Mentik dari notebook.")
    return model, device, Lock()


def read_image(raw):
    with Image.open(BytesIO(raw)) as opened:
        if opened.width * opened.height > MAX_PIXELS:
            raise ValueError("Resolusi gambar melebihi 24 megapiksel. Perkecil gambar lalu unggah kembali.")
        return ImageOps.exif_transpose(opened).convert("RGB")


def predict(model_name, model, device, lock, image, confidence, iou):
    import torch

    with lock, torch.inference_mode():
        if model_name == "YOLOv5s":
            model.conf = confidence
            model.iou = iou
            model.max_det = MAX_DETECTIONS
            predictions = model(image, size=IMAGE_SIZE)
            boxes = predictions.xyxy[0].detach().cpu().tolist()
            names = predictions.names
        else:
            result = model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=confidence,
                iou=iou,
                max_det=MAX_DETECTIONS,
                device=device,
                verbose=False,
            )[0]
            names = result.names
            boxes = [
                [*xyxy, score, class_id]
                for xyxy, score, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                )
            ]
    return [
        (str(names[int(class_id)]).strip(), float(score), (x1, y1, x2, y2))
        for x1, y1, x2, y2, score, class_id in boxes
    ]


def annotate(image, detections):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    line = max(2, round(max(image.size) / 450))
    font = ImageFont.load_default(size=max(13, min(48, line * 5)))

    for name, confidence, box in reversed(detections):
        x1, y1, x2, y2 = [
            max(0, min(bound - 1, round(coordinate)))
            for coordinate, bound in zip(box, (*image.size, *image.size))
        ]
        color = COLORS[name.lower()]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
        label = f"{name} {confidence:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left + 8
        label_height = bottom - top + 6
        x = min(x1, max(0, image.width - label_width))
        y = max(0, y1 - label_height)
        draw.rectangle((x, y, x + label_width, y + label_height), fill=color)
        draw.text((x + 4 - left, y + 3 - top), label, fill="#101820", font=font)
    return output


def main():
    st.set_page_config(page_title="Deteksi Benih Padi", page_icon="🌾", layout="wide")
    st.title("Deteksi Benih Padi")
    st.write("Unggah foto benih padi untuk mendeteksi varietas Ciherang, IR64, dan Mentik.")

    with st.sidebar:
        st.header("Pengaturan")
        available = [name for name, path in MODEL_FILES.items() if path.is_file()]
        model_options = list(MODEL_FILES)
        default_model = model_options.index(available[0]) if available else 0
        model_name = st.selectbox("Model", model_options, index=default_model)
        confidence = st.slider("Ambang confidence", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("Ambang IoU untuk NMS", 0.10, 0.90, 0.45, 0.05)
        st.caption("Ukuran inferensi 800 piksel. Maksimum 1.000 deteksi per gambar.")

    checkpoint = MODEL_FILES[model_name]
    if not checkpoint.is_file():
        st.warning(f"Model {model_name} memerlukan file `models/{checkpoint.name}`. Lihat README.")
    upload = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        st.session_state.pop("detection_result", None)
        return

    raw = upload.getvalue()
    if len(raw) > 20 * 1024 * 1024:
        st.error("Ukuran file melebihi 20 MB.")
        return
    try:
        image = read_image(raw)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        st.error("Gambar gagal dibaca. Gunakan file JPG, PNG, atau WEBP yang valid, maksimal 24 megapiksel.")
        return

    original, output = st.columns(2)
    original.image(image, caption="Gambar asli")
    stamp = None
    if checkpoint.is_file():
        info = checkpoint.stat()
        stamp = (info.st_mtime_ns, info.st_size)
    result_key = (sha256(raw).hexdigest(), model_name, confidence, iou, stamp)
    saved = st.session_state.get("detection_result")
    if saved is not None and saved["key"] != result_key:
        del st.session_state["detection_result"]

    if st.button("Deteksi gambar", type="primary", disabled=stamp is None):
        st.session_state.pop("detection_result", None)
        with st.spinner(f"Menjalankan {model_name}…"):
            try:
                model, device, lock = load_model(model_name, str(checkpoint), stamp)
                detections = predict(model_name, model, device, lock, image, confidence, iou)
                buffer = BytesIO()
                annotate(image, detections).save(buffer, format="PNG")
                st.session_state["detection_result"] = {
                    "key": result_key,
                    "png": buffer.getvalue(),
                    "detections": detections,
                }
            except Exception as exc:
                st.error(f"Deteksi gagal. {exc}")
                return

    saved = st.session_state.get("detection_result")
    if saved is None:
        return
    detections = saved["detections"]
    output.image(saved["png"], caption=f"Hasil deteksi {model_name}")
    st.metric("Total objek terdeteksi", len(detections))
    if detections:
        st.dataframe(
            [
                {"No.": index, "Varietas": name, "Confidence": f"{score:.1%}"}
                for index, (name, score, _) in enumerate(detections, start=1)
            ],
            hide_index=True,
        )
    else:
        st.info("Tidak ada objek yang lolos ambang confidence untuk gambar ini.")
    st.download_button(
        "Unduh gambar hasil deteksi",
        data=saved["png"],
        file_name=f"{Path(upload.name).stem}_deteksi.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
