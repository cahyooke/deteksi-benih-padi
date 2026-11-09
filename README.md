# Deteksi Benih Padi

Aplikasi Streamlit untuk mengunggah gambar dan mendeteksi benih Ciherang, IR64, dan Mentik. Bobot YOLOv5s dari run rice_yolov5s_v2 sudah tersedia di `models/yolov5s_best.pt`.

## Unggah ke GitHub

1. Ekstrak paket ZIP.
2. Buat repository GitHub bernama `deteksi-benih-padi`. Pilih Private untuk menyimpan bobot riset dalam repo pribadi. Aktifkan Add a README file agar repository mempunyai branch main.
3. Di repository, pilih **Add file → Upload files**.
4. Buka folder hasil ekstrak. Seret isi folder ke halaman unggah. Pastikan `app.py`, `requirements.txt`, `packages.txt`, `models/`, dan `yolov5/` berada di tingkat teratas repository.
5. Klik **Commit changes**.

## Jalankan di Streamlit Community Cloud

1. Buka https://share.streamlit.io/ dan masuk dengan akun GitHub yang punya akses ke repository.
2. Pilih **Create app** dan opsi untuk memakai repository yang sudah ada.
3. Pilih repository `deteksi-benih-padi`, branch `main`, dan **Main file path** `app.py`.
4. Pada **Advanced settings**, pilih Python 3.12, kemudian pilih **Deploy**.
5. Setelah aplikasi tampil, pilih YOLOv5s, unggah foto, lalu klik **Deteksi gambar**.

Pilihan YOLO11n memerlukan bobot `models/yolo11n_best.pt` dari run YOLO11n Anda.

## Sumber model

Bobot YOLOv5s berasal dari folder Drive riset yang dibagikan. Salinan di sini memakai metadata jalur yang portabel. Data tensor bobot sama dengan checkpoint asli.

Modul inferensi YOLOv5 berasal dari commit `cb2f17fd5d790cfdd0c07c25351904df9affbe6e` pada https://github.com/ultralytics/yolov5. Lisensi sumber tersedia di `yolov5/LICENSE`.
