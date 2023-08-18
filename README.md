# Laporan Proyek Machine Learning - Sitti Khairunnisak
Machine learning Klasifikasi Gambar, menentukan gambar kucing atau anjing

## Domain Proyek
Hewan merupakan makhluk hidup selalu ada di sekitar kita. Masyarakatpun banyak yang memelihara hewan peliharaan terutama anjing dan kucing, karena memiliki karakter dan fungsi yang
beragam dan menyenangkan manusia. Pada pengolahan citra, proses pengklasifikasian objek merupakan salah satu bagian permasalahan dalam computer Vision. Tujuan pengklasifikasian citra ini
adalah proses memasukkan citra kedalam beberapa kategori yang disesuaikan dengan kebutuhan. Ide dari pengklasifikasian citra yang spesifik dengan memberi masukkan dari sekumpulan angka 
yang diproses dan menghasilkan angka yang merupakan representasi dari kategori citra tersebut, dan hasil dari klasifikasi citra digital dapat menjadi alternatif dalam mengenali hewan.
Selain itu proses mengklasifikasikan citra anjing dan kucing ini diharapkan adalah komputer dapat mengenali dan membedakan objek pada citra selayaknya manusia.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Penyelesaian menentukan citra atau objen kucing dan anjing tersebut dengan cara membuat program dengan cnn yang dapat mengenali bentuk hewan kucing dan anjing.
- Penambahan dataset yang digunakan untuk training dengan resolusi gambar yang bagus dapat membuat model yang dipakai lebih baik dan mengurangi overfitting
  
  Format Referensi:  [Klasifikasi citra anjing dan kucing menggunakan metode convolutional neural network (CNN)]
  (Riyadi, A. S., Wardhani, I. P., & Widayati, S. (2021, September). Klasifikasi citra anjing dan kucing menggunakan metode convolutional neural network (CNN). In Prosiding Seminar SeNTIK (Vol. 5, No. 1, pp. 307-311).) 

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- bagaimana proses klasifikasi dengan CNN mampu menghasilkan deteksi objek citra anjing dan kucing serta membedakan modelnya?
- Bagaimana penggunaan dataset training dengan resolusi citra (gambar) yang bagus dapat membuat model yang dipakai lebih baik dan mengurangi overfitting?
- Berapa % capaian akurasi dan presisi sistem klasifikasi terhadap penganalan anjing dan kucing ?


### Goals

Menjelaskan tujuan dari pernyataan masalah:
- langkah-langkah yaitu import beberapa libraries yang dapat mendeteksi gambar. Lalu mendefinisikan ukuran gambar yang ingin diterapkan. Selanjutnya menggunakan
dataset dalam mengkategorikan anjing dan kucing. Selanjutnya penulis menggunakan dataset yang telah diperoleh untuk mendapatkan data yang dapat dilatih dari
total data yang didapatkan

![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/7f1e9a81-16f0-48ab-80a4-7327a2cc5072)

- Menvisualisasikan grafik keakuratan pada proses training hingga validasi data yang diperoleh. Dan menggunakan Callback untuk mengurangi overftting
- % capaian akurasi berhenti ketika 80 sesuai target callback
- ![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/51278977-3994-4f31-8657-30ed102f43ef)
- 

## Data Understanding
data yang digunakan mengimport dari kaggle https://www.kaggle.com/datasets/tongpython/cat-and-dog

### Variabel-variabel pada cats and dog dataset adalah sebagai berikut:
- cats : merupakan hewan jenis kucing
- dogs : merupakan hewan jenis anjing

## Data Preparation
1. Mengambil beberapa libraries yang
dapat mendeteksi gambar.
2. mengimport dataset dari google drive
3. menentukan direktori
4. menampilkan jumlah gambar
5. Menggunakan dataset untuk
mengkategorikan anjing dan kucing.
6. Menkonversikan kolom kategori
menjadi tipe data string.
7. Menerapkan image generator dalam
proses klasifikasi dengan menggunakan
gambar
8. menggunakan model cnn
9. menentukan callback
10. menentukan model fit
11. Menvisualisasikan grafik keakuratan
pada proses training hingga validasi
data yang diperoleh.


## Modeling
menetukan model cnn
 #Membentuk model sequential
Model = tf.keras.models.Sequential([                                                
membentuk model summary
dan melakukan komplikasi model
model.compile(loss='binary_crossentropy',                                         #Loss function Yang digunakan untuk Klasifikasi binary tidak Lebih Dari 2
              optimizer=tf.optimizers.Adam(),                                     #Fungsi optimizer 'adam'
              metrics=['accuracy'])                                               #Menampilkan akurasi model training

## Evaluation
evaluasi akurasi dengan menentukan nilai akurasi training dan validasi

**---Ini adalah bagian akhir laporan---**
