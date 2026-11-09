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
    with Image.open(BytesIO(raw)) as opened:
        if opened.width * opened.height > MAX_PIXELS:
            raise ValueError("Resolusi gambar melebihi 24 megapiksel. Perkecil gambar lalu unggah kembali.")
        return ImageOps.exif_transpose(opened).convert("RGB")


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


STYLES = """
<style>
[data-testid="stMainBlockContainer"] {max-width: 65rem; padding-top: 3.4rem;}
.seed-masthead {position: relative; overflow: hidden; isolation: isolate;
  padding: 2rem clamp(1.25rem,5vw,3rem); border-radius: 1.15rem;
  background: #172842; color: #fff8ed; margin-bottom: 1rem;}
.seed-masthead h1 {color: #fff8ed; font-family: Georgia, serif;
  font-size: clamp(2.1rem,5vw,3.6rem); line-height: 1.06; padding: .45rem 0 .65rem;}
.seed-masthead p {max-width: 36rem; margin: 0; color: #e6edf6;}
.seed-eyebrow {color: #ffd494; font-size: .75rem; font-weight: 750; letter-spacing: .14em;}
.seed-mark {position: absolute; right: 1.4rem; top: 50%; width: 10rem;
  transform: translateY(-50%); opacity: .75; z-index: -1; pointer-events: none;}
.seed-step {color: #8a4b25; font-size: .72rem; font-weight: 750; letter-spacing: .12em;}
.seed-status {border-left: 4px solid #bd7138; padding: .8rem 1rem;
  border-radius: 0 .5rem .5rem 0; background: #f3e8d7; color: #25354c;}
.seed-status p {margin: .25rem 0 0;}
@media (max-width: 800px) {
  .seed-mark {opacity: .24;}
}
</style>
"""

MASTHEAD = """
<header class="seed-masthead">
  <div class="seed-eyebrow">LAB CITRA PADI</div>
  <h1>Deteksi Benih Padi</h1>
  <p>Periksa lokasi butir dan prediksi varietas Ciherang, IR64, dan Mentik dari satu foto.</p>
  <svg class="seed-mark" viewBox="0 0 160 160" aria-hidden="true" focusable="false">
    <circle cx="80" cy="80" r="72" fill="none" stroke="#b98954" stroke-width="1"/>
    <path d="M37 123c31-25 57-46 85-83" fill="none" stroke="#f3c582" stroke-width="2"/>
    <g fill="#f3c582">
      <path d="M49 106c-14-3-22-14-20-27 12 0 21 10 20 27Z"/>
      <path d="M62 95c4-14 17-23 31-19-4 13-16 21-31 19Z"/>
      <path d="M72 87c-13-5-19-19-15-31 13 4 19 16 15 31Z"/>
      <path d="M84 76c5-15 18-22 31-18-5 13-18 21-31 18Z"/>
      <path d="M93 65c-10-6-13-19-8-30 11 5 15 17 8 30Z"/>
      <path d="M107 53c1-14 12-25 26-25-1 13-11 25-26 25Z"/>
    </g>
  </svg>
</header>
"""


def render_guide():
    st.subheader("Panduan penggunaan")
    with st.container(border=True):
        st.markdown(
            "1. Gunakan foto tajam dengan pencahayaan merata dan latar yang polos. "
            "Pastikan butir terlihat jelas.\n"
            "2. Buka Pengaturan analisis untuk menyesuaikan ambang confidence bila diperlukan.\n"
            "3. Unggah JPG, PNG, atau WebP, lalu pilih **Jalankan deteksi**.\n"
            "4. Periksa kotak, label varietas, dan jumlah prediksi. "
            "Anda bisa mengunduh gambar beranotasi dari panel hasil."
        )
    with st.expander("Cara membaca hasil"):
        st.write(
            "Confidence menunjukkan keyakinan model pada tiap prediksi. "
            "Jumlah yang ditampilkan berasal dari prediksi yang lolos ambang tersebut. "
            "Hasil demo pada satu foto berbeda dari metrik akurasi pada dataset uji."
        )
    with st.expander("Pengaturan inferensi"):
        st.write(
            "Naikkan ambang confidence untuk menyaring prediksi berkeyakinan rendah. "
            "IoU NMS mengatur penyaringan kotak deteksi yang bertumpuk."
        )


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


def render_workspace():
    stamp = None
    try:
        info = YOLOV5_WEIGHT.stat()
    except FileNotFoundError:
        pass
    else:
        stamp = (info.st_mtime_ns, info.st_size)

    if stamp is None:
        st.warning("Bobot YOLOv5s belum tersedia di lokasi yang ditentukan.")
    with st.container(border=True):
        st.markdown('<div class="seed-step">01 · MASUKAN</div>', unsafe_allow_html=True)
        st.subheader("Unggah foto")
        upload = st.file_uploader("Foto benih padi", type=["jpg", "jpeg", "png", "webp"])
        st.caption("Gunakan foto tajam, cahaya merata, dan latar polos. Maksimal 20 MB dan 24 megapiksel.")
        with st.expander("Pengaturan analisis"):
            confidence = st.slider("Ambang confidence", 0.05, 0.95, 0.25, 0.05)
            iou = st.slider("Ambang IoU untuk NMS", 0.10, 0.90, 0.45, 0.05)
            st.caption(f"Model {MODEL_NAME} · {IMAGE_SIZE} piksel · maksimum {MAX_DETECTIONS:,} deteksi")

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

    if st.button("Jalankan deteksi", type="primary", disabled=image is None or stamp is None):
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

    st.markdown('<div class="seed-step">02 · KELUARAN</div>', unsafe_allow_html=True)
    st.subheader("Hasil deteksi")
    if saved is None:
        if image is not None:
            st.image(image, caption="Pratinjau gambar asli", width=320)
        st.markdown(
            '<div class="seed-status"><strong>Hasil deteksi akan tampil di sini</strong>'
            '<p>Unggah foto, lalu jalankan deteksi.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        render_result(MODEL_NAME, saved["result"], image, upload.name)


def main():
    st.set_page_config(page_title="Deteksi Benih Padi", page_icon="🌾", layout="wide")
    st.markdown(STYLES, unsafe_allow_html=True)
    st.markdown(MASTHEAD, unsafe_allow_html=True)
    workspace, guide = st.tabs(["Analisis foto", "Panduan penggunaan"])
    with workspace:
        render_workspace()
    with guide:
        render_guide()


if __name__ == "__main__":
    main()
