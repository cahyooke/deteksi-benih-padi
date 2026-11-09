# Deteksi Benih Padi

Aplikasi Streamlit untuk mendeteksi benih padi varietas Ciherang, IR64, dan Mentik pada foto. Aplikasi menampilkan gambar asli, kotak prediksi, jumlah deteksi per varietas, dan unduhan gambar beranotasi.

## Struktur repositori

| Lokasi | Isi |
| --- | --- |
| `app.py` | Antarmuka Streamlit, pemeriksaan gambar, integrasi inferensi, anotasi, dan ekspor hasil |
| `.streamlit/config.toml` | Tema aplikasi dan batas unggah |
| `models/yolov5s_best.pt` | Bobot YOLOv5s dari run `rice_yolov5s_v2` |
| `vendor/yolov5_source.zip` | Salinan kode inferensi Ultralytics YOLOv5 beserta lisensinya |
| `requirements.txt` dan `packages.txt` | Dependensi Python dan sistem untuk deployment |

## Menjalankan aplikasi

Gunakan Python 3.12.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Pada Streamlit Community Cloud, gunakan `app.py` sebagai **Main file path**. Untuk deployment yang sudah menunjuk ke `main` dan `app.py`, gunakan konfigurasi yang sama.

## Panduan penggunaan

1. Unggah foto JPG, PNG, atau WebP. Batas unggah 20 MB dan resolusi maksimal 24 megapiksel.
2. Buka **Pengaturan analisis** di bawah area unggah untuk mengatur ambang confidence dan IoU NMS bila diperlukan.
3. Pilih **Jalankan deteksi** untuk melihat gambar asli, gambar beranotasi, dan jumlah prediksi per varietas.
4. Pilih **Unduh gambar beranotasi** untuk menyimpan hasil. Nilai confidence menunjukkan keyakinan per prediksi.

## Asal komponen

- **Kode aplikasi** berada di `app.py` dan konfigurasi antarmuka di `.streamlit/config.toml`.
- **Bobot terlatih YOLOv5s** berasal dari checkpoint `YOLOv5_Results/rice_yolov5s_v2/weights/best.pt` yang dibagikan rekan penelitian. Semua 358 penyimpanan tensor cocok dengan berkas sumber. Salinan repositori mengubah representasi metadata jalur untuk kompatibilitas. SHA-256 salinan repositori adalah `167c5434d0a88214c17fee237cd1ca663edb5482775166d03b28f97757c42d9a`.
- **Kode YOLOv5 pihak ketiga** berasal dari [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5/tree/cb2f17fd5d790cfdd0c07c25351904df9affbe6e). Arsip menyimpan `LICENSE` GNU AGPL versi 3 dan dokumentasi upstream.
