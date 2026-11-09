# Deteksi Benih Padi dengan Streamlit

Aplikasi ini mengambil bagian inferensi dari `Untitled1.ipynb`. Foto JPG, PNG, atau WEBP bisa diunggah melalui halaman web. Hasilnya menampilkan kotak deteksi, varietas, confidence, dan jumlah objek yang terdeteksi. Gambar hasil dapat diunduh.

## 1. Bobot model

Ekstrak ZIP. Bobot YOLOv5s sudah ada di `models/yolov5s_best.pt`, diambil dari `YOLOv5_Results/rice_yolov5s_v2/weights/best.pt` pada folder Drive yang dibagikan. Salinan dalam paket memakai metadata jalur yang kompatibel lintas sistem operasi. Data tensor bobot diverifikasi sama dengan checkpoint asli.

Jika nanti tersedia bobot YOLO11n dari `YOLOv11_Results/rice_yolo11n_v1/weights/best.pt`, simpan sebagai `models/yolo11n_best.pt`. Folder bobot YOLO11n yang dibagikan belum menampilkan checkpoint saat diperiksa.

Gunakan hanya checkpoint yang Anda percayai karena pemuatan file PyTorch dapat menjalankan kode.

## 2. Pasang dependensi

Buka terminal di folder yang berisi `app.py`. Gunakan Python 3.11 atau 3.12. Contoh instalasi di Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Untuk menjalankan **YOLOv5s yang sudah disertakan**, letakkan repository YOLOv5 di folder `yolov5`, sejajar dengan `app.py`. Perintah berikut mengambil versi repository resmi yang dipakai dalam uji aplikasi:

```powershell
git clone https://github.com/ultralytics/yolov5.git
git -C yolov5 checkout cb2f17fd5d790cfdd0c07c25351904df9affbe6e
.\.venv\Scripts\python.exe -m pip install -r yolov5/requirements.txt
```

Jika Git belum tersedia, unduh ZIP dari [arsip YOLOv5 yang diuji](https://github.com/ultralytics/yolov5/archive/cb2f17fd5d790cfdd0c07c25351904df9affbe6e.zip), ekstrak, lalu ubah nama foldernya menjadi `yolov5`. Jalankan perintah instalasi dependensi YOLOv5 di atas.

YOLO11n dapat langsung dijalankan setelah dependensi utama terpasang dan bobot tersedia.

## 3. Jalankan dan unggah foto

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Di Linux/macOS, gunakan `.venv/bin/python` sebagai pengganti `.\.venv\Scripts\python.exe`. Buka alamat yang tercetak di terminal, biasanya `http://localhost:8501`. Pilih model di panel kiri, unggah satu gambar, lalu klik **Deteksi gambar**. Aplikasi memilih GPU CUDA jika tersedia dan CPU jika tidak.

Ukuran inferensi 800 piksel mengikuti notebook. Ambang awal confidence 0,25 dan IoU NMS 0,45 mengikuti visualisasi prediksi YOLOv5s di notebook dan dipakai sebagai nilai awal tampilan untuk kedua model. Pengaturan ini ditujukan untuk penggunaan per gambar. Untuk percobaan pertama, gunakan foto benih yang menyerupai citra uji notebook.
