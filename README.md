# Analisis Spektrum Matriks Antrean: Universal's Islands of Adventure 🎢

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Array-013243?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Dataframe-150458?style=for-the-badge&logo=pandas)
![Course](https://img.shields.io/badge/IF2123-Aljabar_Linier-red?style=for-the-badge)

> **Tugas Makalah IF2123 Aljabar Linier dan Geometri** > Identifikasi Titik Jenuh Wahana melalui Pendekatan Vektor Eigen dan Rantai Markov.

---

## 📖 Deskripsi Proyek

Proyek ini bertujuan untuk memodelkan dinamika aliran pengunjung di taman hiburan **Universal's Islands of Adventure** menggunakan prinsip Aljabar Linier. Dengan mengambil data waktu tunggu (*wait times*) secara *real-time*, proyek ini menyusun **Matriks Transisi Probabilitas** untuk memprediksi distribusi pengunjung jangka panjang.

Inti dari analisis ini adalah mencari **Vektor Eigen Dominan** ($\lambda = 1$) yang merepresentasikan kondisi **Steady State** (Titik Jenuh) dari sistem antrean, sesuai dengan Teorema Perron-Frobenius.

### 🎯 Tujuan Utama
1.  **Akuisisi Data:** Mengambil data antrean wahana secara otomatis dari API publik.
2.  **Pemodelan Matematis:** Membentuk matriks transisi $5 \times 5$ dari data deret waktu.
3.  **Analisis Spektral:** Menghitung nilai eigen dan vektor eigen untuk menemukan wahana dengan beban antrean terberat.
4.  **Visualisasi:** Membandingkan kondisi aktual vs kondisi stasioner.

---

## ⚙️ Landasan Teori

Sistem antrean dimodelkan sebagai Rantai Markov Diskrit:
$$\mathbf{v}_{t+1} = \mathbf{A}\mathbf{v}_t$$

Dimana:
* $\mathbf{v}_t$: Vektor distribusi probabilitas pengunjung pada waktu $t$.
* $\mathbf{A}$: Matriks transisi stokastik kolom.

Kondisi jenuh (steady state) diperoleh dengan menyelesaikan persamaan eigen:
$$\mathbf{A}\mathbf{x} = \lambda \mathbf{x}$$
dengan $\lambda = 1$.

---

## 📂 Struktur Wahana

Analisis difokuskan pada 5 wahana utama (*Big 5*) di Islands of Adventure:
1.  **Hagrid's Magical Creatures Motorbike Adventure™**
2.  **Jurassic World VelociCoaster**
3.  **The Incredible Hulk Coaster®**
4.  **The Amazing Adventures of Spider-Man®**
5.  **Jurassic Park River Adventure™**

---

## 🚀 Instalasi & Cara Pakai

### 1. Clone Repositori
```bash
git clone [https://github.com/NathanaelGun/MakalahAlgeo.git]
cd MakalahAlgeo