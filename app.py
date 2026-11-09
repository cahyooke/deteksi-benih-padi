from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Lock
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "YOLOv5s"
YOLOV5_WEIGHT = ROOT / "models" / "yolov5s_best.pt"
YOLOV5_SOURCE = ROOT / "vendor" / "yolov5_source.zip"
CLASS_LABELS = {"ciherang": "Ciherang", "ir64": "IR64", "mentik": "Mentik"}
CLASSES = set(CLASS_LABELS)
COLORS = {"ciherang": "#26B9E8", "ir64": "#46D899", "mentik": "#F6BB4B"}
IMAGE_SIZE = 800
MAX_DETECTIONS = 1000
MAX_PIXELS = 24_000_000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@st.cache_resource(show_spinner=False)
def unpack_yolov5():
    """Keep the bundled YOLOv5 source available throughout this process."""
    directory = TemporaryDirectory(prefix="rice_yolov5_")
    with ZipFile(YOLOV5_SOURCE) as archive:
        archive.extractall(directory.name)
    return directory


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_stamp):
    """Cache by checkpoint version so a replacement reloads the model."""
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    source = Path(unpack_yolov5().name)
    model = torch.hub.load(
        str(source), "custom", path=str(YOLOV5_WEIGHT), source="local", device=device
    )
    model.eval()

    names = model.names
    class_names = names.values() if isinstance(names, dict) else names
    if {str(name).strip().lower() for name in class_names} != CLASSES:
        raise ValueError("Bobot YOLOv5s harus mendeteksi Ciherang, IR64, dan Mentik.")
    return model, Lock()


def read_image(raw):
    try:
        with Image.open(BytesIO(raw)) as opened:
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError("Resolusi gambar melebihi 24 megapiksel. Perkecil gambar lalu unggah kembali.")
            return ImageOps.exif_transpose(opened).convert("RGB")
    except ModuleNotFoundError as exc:
        if exc.name != "pi_heif":
            raise
        # Ultralytics may try an optional HEIF decoder after Pillow rejects a file.
        raise ValueError("Gunakan gambar JPG, PNG, atau WebP yang valid.") from exc


def predict(model, lock, image, confidence, iou):
    import torch

    with lock, torch.inference_mode():
        model.conf = confidence
        model.iou = iou
        model.max_det = MAX_DETECTIONS
        predictions = model(image, size=IMAGE_SIZE)
        boxes = predictions.xyxy[0].detach().cpu().tolist()
        names = predictions.names
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
        label = f"{CLASS_LABELS[name.lower()]} {confidence:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left + 8
        label_height = bottom - top + 6
        x = min(x1, max(0, image.width - label_width))
        y = max(0, y1 - label_height)
        draw.rectangle((x, y, x + label_width, y + label_height), fill=color)
        draw.text((x + 4 - left, y + 3 - top), label, fill="#101820", font=font)
    return output


def render_result(model_name, result, image, upload_name):
    if "error" in result:
        st.error(f"{model_name} gagal dijalankan. {result['error']}")
        return

    detections = result["detections"]
    counts = Counter(name.lower() for name, _, _ in detections)
    total, *varieties = st.columns(4)
    total.metric("Total prediksi", len(detections))
    for column, (key, label) in zip(varieties, CLASS_LABELS.items()):
        column.metric(label, counts[key])

    original, annotated = st.columns(2)
    original.image(image, caption="Gambar asli")
    annotated.image(result["png"], caption=f"Hasil deteksi · {model_name}")

    if detections:
        with st.expander("Lihat detail prediksi"):
            st.dataframe(
                [
                    {"No.": index, "Varietas": CLASS_LABELS[name.lower()],
                     "Confidence": f"{score:.1%}"}
                    for index, (name, score, _) in enumerate(detections, start=1)
                ],
                hide_index=True,
            )
    else:
        st.info("Tidak ada objek yang lolos ambang confidence pada foto ini.")

    st.download_button(
        "Unduh gambar beranotasi",
        data=result["png"],
        file_name=f"{Path(upload_name).stem}_{model_name}_deteksi.png",
        mime="image/png",
        key=f"download-{model_name}",
    )


HEADER = """
<style>
[data-testid="stMainBlockContainer"] {
    max-width: 62rem;
    padding-top: clamp(2.75rem, 5vw, 4.25rem);
}
.rice-header {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    padding: clamp(1.25rem, 4vw, 2.25rem);
    margin-bottom: .6rem;
    border: 1px solid #d5e2f2;
    border-top: 4px solid #315b91;
    border-radius: .9rem;
    background: linear-gradient(118deg, #f3f8ff 32%, #fbfcff 100%);
    color: #182b46;
}
.rice-header h1 {
    padding: .25rem 0 .5rem;
    font-size: clamp(1.8rem, 5vw, 2.8rem);
    line-height: 1.12;
    color: #182b46;
}
.rice-header p { margin: 0; max-width: 35rem; color: #344d6c; }
.rice-kicker { font-size: .78rem; font-weight: 700; letter-spacing: .09em; color: #355984; }
.rice-varieties { display: flex; gap: .45rem; flex-wrap: wrap; margin-top: 1rem; }
.rice-varieties span {
    border: 1px solid #d5e2f2;
    border-radius: 999px;
    padding: .2rem .65rem;
    background: #ffffff;
    color: #243f62;
    font-size: .85rem;
}
.rice-pattern {
    position: absolute;
    inset: 0 0 0 auto;
    z-index: -1;
    width: 29%;
    background: repeating-linear-gradient(135deg, transparent 0 15px, #dce8f8 15px 16px);
    mask-image: linear-gradient(to right, transparent, black);
}
@media (max-width: 640px) {
    .rice-pattern { opacity: .45; }
}
</style>
<header class="rice-header">
    <div class="rice-pattern" aria-hidden="true"></div>
    <div class="rice-kicker">DETEKSI CITRA &middot; YOLOv5s</div>
    <h1>Deteksi Benih Padi</h1>
    <p>Unggah foto benih padi untuk melihat lokasi dan varietas setiap butir yang terdeteksi.</p>
    <div class="rice-varieties">
        <span>Ciherang</span><span>IR64</span><span>Mentik</span>
    </div>
</header>
"""


def main():
    st.set_page_config(page_title="Deteksi Benih Padi", page_icon="🌾", layout="centered")
    st.markdown(HEADER, unsafe_allow_html=True)

    stamp = None
    try:
        info = YOLOV5_WEIGHT.stat()
    except FileNotFoundError:
        pass
    else:
        stamp = (info.st_mtime_ns, info.st_size)

    if stamp is None:
        st.warning("Bobot YOLOv5s belum tersedia di lokasi yang ditentukan.")
    input_panel = st.container(border=True)
    with input_panel:
        st.subheader("Foto untuk dianalisis")
        upload = st.file_uploader("Unggah foto benih padi", type=["jpg", "jpeg", "png", "webp"])
        st.caption("Pilih foto tajam dengan latar polos. Maksimal 20 MB dan 24 megapiksel.")
        with st.expander("Pengaturan deteksi"):
            confidence = st.slider(
                "Ambang confidence", 0.05, 0.95, 0.25, 0.05,
                help="Naikkan ambang untuk menyaring prediksi dengan keyakinan rendah.",
            )
            iou = st.slider(
                "Ambang IoU untuk NMS", 0.10, 0.90, 0.45, 0.05,
                help="Mengatur penyaringan kotak deteksi yang saling bertumpuk.",
            )
            st.caption(f"Ukuran inferensi {IMAGE_SIZE} piksel.")

    image = None
    raw = None
    if upload is None:
        st.session_state.pop("detection_result", None)
    else:
        raw = upload.getvalue()
        if len(raw) > MAX_UPLOAD_BYTES:
            st.error("Ukuran file melebihi 20 MB.")
        else:
            try:
                image = read_image(raw)
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
                message = str(exc) if isinstance(exc, ValueError) else "Gunakan gambar JPG, PNG, atau WebP yang valid."
                st.error(f"Gambar gagal dibaca. {message}")

    key = (sha256(raw).hexdigest(), stamp, confidence, iou) if image is not None else None
    saved = st.session_state.get("detection_result")
    if saved is not None and saved["key"] != key:
        st.session_state.pop("detection_result", None)
        saved = None

    if input_panel.button("Jalankan deteksi", type="primary", disabled=image is None or stamp is None):
        st.session_state.pop("detection_result", None)
        with st.spinner(f"Menjalankan {MODEL_NAME}…"):
            try:
                model, lock = load_model(stamp)
                detections = predict(model, lock, image, confidence, iou)
                buffer = BytesIO()
                annotate(image, detections).save(buffer, format="PNG")
                result = {"png": buffer.getvalue(), "detections": detections}
            except Exception as exc:
                result = {"error": str(exc)}
        saved = {"key": key, "result": result}
        st.session_state["detection_result"] = saved

    if saved is not None:
        with st.container(border=True):
            st.subheader("Hasil deteksi")
            render_result(MODEL_NAME, saved["result"], image, upload.name)
    elif image is not None:
        st.image(image, caption="Pratinjau foto", width=320)


if __name__ == "__main__":
    main()
