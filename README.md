# Laporan Proyek Machine Learning - Sitti Khairunnisak
Machine learning Klasifikasi Gambar, menentukan gambar kucing atau anjing

## Domain Proyek

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


- Menvisualisasikan grafik keakuratan pada proses training hingga validasi data yang diperoleh. Dan menggunakan Callback untuk mengurangi overftting
- % capaian akurasi berhenti ketika 80 sesuai target callback

## Data Understanding
data yang digunakan mengimport dari kaggle, dataset terdiri dari dua kategori, yaitu anjing dan kucing. kategori kucing ada 1011 gambar dan untuk kategori anjing ada 1015 gambar
https://www.kaggle.com/datasets/tongpython/cat-and-dog



### Variabel-variabel pada cats and dog dataset adalah sebagai berikut:
- cats : merupakan hewan jenis kucing
- dogs : merupakan hewan jenis anjing

## Data Preparation
Mengapa menggunakan image generator?
Karena Image generator digunakan untuk membuat gambar dari teks atau input data lainnya yang dapat membantu dalam pemrosesan data. Image generator dapat membantu masalah pengolahan data, mampu membuat membuat gambar tambahan dari yang sudah ada, dan dapat membantu melatih model pembelajaran mesin. Dengan menghasilkan gambar baru model dapat dilatih pada kumpulan data yang lebih besar dan beragam, yang dapat meningkatkan akurasinya. image generator juga dapat digunakan untuk membuat representasi visual dari data, seperti grafik. kita bisa memanggilnya dengan cara import imagedatagenerator seperti contoh kode ini, 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
Dan nantinya ukuran data bisa dirubah, seperti contoh kode dibawah
train_datagen = ImageDataGenerator(
    rescale = 1./255, 
    rotation_range = 20,                                                   
    shear_range = 0.2,                                                           
    fill_mode = 'nearest',                                                       


## Modeling
membuat model cnn
 #Membentuk model sequential
Model = tf.keras.models.Sequential([   
tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape= (150,150,3)),  membuat 3 layer convolusional yang digunakan untuk memproses input data yang nantinya akan diekstrak ke dalam matriks, menggunakan fungsi activasi relu untuk memproses inputan yang diinput fungsi convisional, dan bentuk input, gambar dengan ukuran 150x150 dengan 3 byte, dan membuat layer maxpooling
tf.keras.layers.MaxPooling2D(2,2),                           
tf.keras.layers.Flatten(),  digunakan untuk membuat data flat dan digunakan untuk input data selanjutnya
tf.keras.layers.Dropout(0.5), layer dropout untuk mencegah terjadinya overfitting
tf.keras.layers.Dense(64, activation= 'relu'), 
layer play menggunakan hidden layer dengan fungsi activation relu
tf.keras.layers.Dense(2, activation= 'sigmoid')                                   layer output menggunakan fungsi sigomid dengan banyak output 2 
membentuk model summary
dan melakukan komplikasi model menggunakanfungsi optimizer adam yang digunakan untuk mengupdate iterasinya supaya lebih cepat mencapai titik yang lebih optimal
model.compile(loss='binary_crossentropy',                                        
              optimizer=tf.optimizers.Adam(),                                     
              metrics=['accuracy'])                                              

## Evaluation
Tujuan visualisasi ini adalah untuk membantu memahami bagaimana model berkembang selama pelatihan. Kita dapat mengamati apakah model cenderung overfit atau underfit, serta mengkaji tren akurasi dan loss untuk setiap epoch. Visualisasi ini juga dapat membantu dalam pemilihan parameter dan pengambilan keputusan terkait dengan model.

**---Ini adalah bagian akhir laporan---**
