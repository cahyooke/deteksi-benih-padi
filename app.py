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
[data-testid="stMainBlockContainer"] {max-width: 78rem; padding-top: 3.7rem;}
[data-testid="stSidebar"] {border-right: 1px solid #dce7df;}
.rice-brand {padding: .85rem 1rem; border-left: 4px solid #277459;
  border-radius: .65rem; background: #dcebe2; color: #193228; margin-bottom: .85rem;}
.rice-brand strong {font-size: 1.02rem;}
.rice-brand small {display: block; margin-top: .15rem; color: #3b6254;}
.rice-hero {padding: 1.8rem 2rem; border: 1px solid #cde1d4;
  border-radius: 1rem; background: linear-gradient(115deg,#deeee4,#f1f6ed);
  color: #193228; margin-bottom: .85rem;}
.rice-hero h1 {color: #193228; font-size: clamp(1.85rem,4vw,2.85rem);
  line-height: 1.12; padding: .35rem 0 .55rem;}
.rice-hero p {max-width: 44rem; color: #3b6254; margin-bottom: .7rem;}
.rice-kicker {font-size: .73rem; font-weight: 750; letter-spacing: .13em;}
.rice-chip {display: inline-block; font-size: .8rem; font-weight: 650;
  margin: .2rem .32rem 0 0; padding: .25rem .65rem; border-radius: 2rem;
  border: 1px solid #c0dac9; background: #f5faf6;}
.rice-empty {text-align: center; padding: 1.65rem 1rem; border-radius: .8rem;
  border: 1px dashed #aac8b6; background: #eaf2ec; color: #193228;}
.rice-empty p {margin: .4rem 0 0; color: #3b6254;}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] {padding-top: 3rem;}
  .rice-hero {padding: 1.25rem;}
}
</style>
"""


def render_guide():
    st.subheader("Panduan penggunaan")
    with st.container(border=True):
        st.markdown(
            "1. Gunakan foto tajam dengan pencahayaan merata dan latar yang polos. "
            "Pastikan butir terlihat jelas.\n"
            "2. Periksa model YOLOv5s dan atur ambang confidence di panel samping.\n"
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


def main():
    st.set_page_config(page_title="Deteksi Benih Padi", page_icon="🌾", layout="wide")
    st.markdown(STYLES, unsafe_allow_html=True)
    stamp = None
    try:
        info = YOLOV5_WEIGHT.stat()
    except FileNotFoundError:
        pass
    else:
        stamp = (info.st_mtime_ns, info.st_size)

    with st.sidebar:
        st.markdown(
            '<div class="rice-brand"><strong>Deteksi Benih Padi</strong>'
            '<small>Ruang inspeksi varietas</small></div>',
            unsafe_allow_html=True,
        )
        page = st.radio("Ruang kerja", ["Deteksi gambar", "Panduan penggunaan"])
        st.divider()
        st.subheader("Konfigurasi")
        st.markdown(f"**Model aktif**  \n{MODEL_NAME}")
        confidence = st.slider("Ambang confidence", 0.05, 0.95, 0.25, 0.05)
        with st.expander("Pengaturan inferensi"):
            iou = st.slider("Ambang IoU untuk NMS", 0.10, 0.90, 0.45, 0.05)
            st.caption(f"Ukuran inferensi {IMAGE_SIZE} piksel. Maksimum 1.000 deteksi per gambar.")

    st.caption(f"Ruang kerja / {page} · {MODEL_NAME}")
    st.markdown(
        '<section class="rice-hero"><div class="rice-kicker">INSPEKSI CITRA</div>'
        '<h1>Deteksi Benih Padi</h1>'
        '<p>Temukan lokasi butir dan lihat prediksi varietas pada foto yang Anda unggah.</p>'
        '<span class="rice-chip">Ciherang</span><span class="rice-chip">IR64</span>'
        '<span class="rice-chip">Mentik</span></section>',
        unsafe_allow_html=True,
    )
    if page == "Panduan penggunaan":
        render_guide()
        return

    if stamp is None:
        st.warning("Bobot YOLOv5s belum tersedia di lokasi yang ditentukan.")
    uploader, tips = st.columns([3, 2], gap="large")
    with uploader:
        with st.container(border=True):
            st.subheader("Gambar untuk dianalisis")
            upload = st.file_uploader("Unggah foto benih padi", type=["jpg", "jpeg", "png", "webp"])
            st.caption("JPG, PNG, atau WebP · maksimal 20 MB dan 24 megapiksel")
    with tips:
        st.subheader("Siapkan foto Anda")
        st.write("Pilih foto tajam dengan cahaya merata dan latar polos. Pisahkan butir agar kotak hasilnya mudah diperiksa.")
        st.caption("Cara membaca confidence dan pengaturan tersedia di menu Panduan penggunaan.")

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

    st.subheader("Hasil inspeksi")
    if saved is None:
        if image is not None:
            st.image(image, caption="Pratinjau gambar asli", width=320)
        st.markdown(
            '<div class="rice-empty"><strong>Hasil deteksi akan tampil di sini</strong>'
            '<p>Unggah foto, lalu jalankan deteksi.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        render_result(MODEL_NAME, saved["result"], image, upload.name)


if __name__ == "__main__":
    main()
