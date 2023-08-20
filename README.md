# Laporan Proyek Machine Learning - Sitti Khairunnisak
Machine learning Klasifikasi Gambar, menentukan gambar kucing atau anjing

## Domain Proyek
Hewan merupakan makhluk hidup selalu ada di sekitar kita. Masyarakatpun banyak yang memelihara hewan peliharaan terutama anjing dan kucing, karena memiliki karakter dan fungsi yang beragam dan menyenangkan manusia. Pada pengolahan citra, proses pengklasifikasian objek merupakan salah satu bagian permasalahan dalam _computer Vision_. Tujuan pengklasifikasian citra ini adalah proses memasukkan citra kedalam beberapa kategori yang disesuaikan dengankebutuhan. Ide dari pengklasifikasian citra yang spesifik dengan memberi masukkan dari sekumpulan angka yang diproses dan menghasilkan angka yang merupakan representasi dari kategori citra tersebut, dan hasil dari klasifikasi citra digital dapat menjadi alternatif dalam mengenali hewan. Selain itu proses mengklasifikasikan citra anjing dan kucing ini diharapkan adalah komputer dapat mengenali dan membedakan objek pada citra selayaknya manusia[1]
Pengenalan wajah menggunakan machine learning sangat penting. Dengan menggunakan machine learning, teknologi pengenalan wajah dapat mencapai tingkat akurasi yang tinggi dalam mengenali wajah hewan (kucing dan anjing) atau wajah seseorang. Algoritma machine learning seperti _neural network_ dapat meniru proses otak manusia dalam mengenali fitur-fitur khusus pada wajah, seperti jarak antara mata, tinggi dahi, lebar hidung, dan sebagainya. Algoritma _facial recognition_ dirancang untuk memetakan fitur wajah seseorang secara matematis.
selain itu Teknologi pengenalan wajah menggunakan machine learning dapat diterapkan dalam berbagai bidang, seperti keamanan, pengenalan identitas, sehingga dapat digunakan untuk mendeteksi ancaman dan memprediksi risiko keamanan.
Referensi: (gambar 1)  ![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/b329010f-6ae9-4c7c-9346-b9f87c29fd10) 
## Business Understanding
apa dampak positif dari pemecahan masalah pengenalan wajah hewan pada  pemilik hewan peliharaan?
dampak positifnya adalah adalah dapat memudahkan dalam memantau aktivitas anjing atau kucing mereka secara real-time. Hal ini dapat membantu pemilik hewan peliharaan untuk memastikan bahwa hewan peliharaan mereka aman dan tidak melakukan hal-hal yang tidak diinginkan, dan juga meningkatkan keamanan, jadi pemilik hewan peliharaan dapat memastikan bahwa hanya anjing atau kucing mereka yang dapat masuk ke dalam rumah atau area tertentu. Hal ini dapat membantu mencegah anjing atau kucing yang tidak diinginkan masuk ke dalam rumah dan mengganggu hewan peliharaan yang ada di dalamnya. dan dengan penerapan teknologi pengenalan otomatis menggunakan machine learning, diharapkan dapat memberikan kemudahan, keamanan, dan kenyamanan bagi pemilik hewan peliharaan dalam merawat dan memantau aktivitas hewan peliharaan mereka.

### Problem Statements
permasalahan yang ada dalam proyek pengenalan wajah kucing dan anjing adalah
- bagaimana proses klasifikasi dengan CNN mampu menghasilkan deteksi objek citra anjing dan kucing serta membedakan modelnya?
- Bagaimana penggunaan dataset training dengan resolusi citra (gambar) yang bagus dapat membuat model yang dipakai lebih baik dan mengurangi overfitting?
- Berapa % capaian akurasi dan presisi sistem klasifikasi terhadap penganalan anjing dan kucing ?

### Goals
pembahasan mengenai penyelesaian masalah diatas adalah dengan
- langkah-langkah yaitu import beberapa libraries yang dapat mendeteksi gambar. Lalu mendefinisikan ukuran gambar yang ingin diterapkan. Selanjutnya menggunakan
dataset dalam mengkategorikan anjing dan kucing. Selanjutnya penulis menggunakan dataset yang telah diperoleh untuk mendapatkan data yang dapat dilatih dari
total data yang didapatkan
- Menvisualisasikan grafik keakuratan pada proses training hingga validasi data yang diperoleh. Dan menggunakan Callback untuk mengurangi overftting
- pencapaian % akurasi berhenti ketika 80 sesuai target callback

## Data Understanding
data yang digunakan mengimport dari kaggle, https://www.kaggle.com/datasets/tongpython/cat-and-dog
(gambar 2) ![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/118ed360-c953-45b6-ab08-ebac7351e16f)
setelah menentukan direktori, dari isi folder bahan, latih dan validasi, kita print jumlah dataset yang terdiri dari dua kategori, yaitu anjing dan kucing. kategori kucing ada 1011 gambar dan untuk kategori anjing ada 1015 gambar    
Dataset terdiri dari sejumlah gambar kucing dan anjing. Jumlah pasti gambar dalam kumpulan data dapat bervariasi dan bergantung pada ukuran kumpulan data yang digunakan. 
Setiap data dalam kumpulan data direpresentasikan dalam format file gambar seperti JPEG atau PNG. Setiap gambar memiliki ukuran dan resolusi yang berbeda.
contoh visualisasi kucing ini merupakan bagian dari evaluation.
(gambar 3)![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/70e72744-0d42-4959-9393-489768df685d)

### Variabel-variabel pada cats and dog dataset adalah sebagai berikut:
Variabel dalam kumpulan data adalah gambar yang berupa kucing atau anjing. Gambar tersebut digunakan untuk melatih dan menguji algoritme pembelajaran mesin untuk mengklasifikasikan apakah suatu gambar berisi kucing atau anjing
- cats : merupakan hewan jenis kucing
- dogs : merupakan hewan jenis anjing

## Data Preparation
Mengapa menggunakan_ image generator_?
Karena _Image generator_ digunakan untuk membuat gambar dari teks atau input data lainnya yang dapat membantu dalam pemrosesan data. _Image generator_ dapat membantu masalah pengolahan data, mampu membuat gambar tambahan dari yang sudah ada. Dengan menghasilkan gambar baru model dapat dilatih pada kumpulan data yang lebih besar dan beragam, yang dapat meningkatkan akurasinya. Langkah pertama adalah mengimpor pustaka yang diperlukan dengan cara import _imagedatagenerator._
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/22bff0f8-0cf4-4811-b374-b494ffc9dda8)
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/6609b03c-7993-4908-89a9-1971de3a7c18)
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/ae1628be-a4bd-4fbd-8f4b-3e760bec2251)
tensorflow untuk membuat dan melatih model.
ImageDataGenerator dari tensorflow.keras.preprocessing.image untuk augmentasi data dan menyiapkan generator data untuk pelatihan dan validasi.
dan menggunakan _ImageDataGenerator _untuk melakukan augmentasi data pada gambar pelatihan. Beberapa augmentasi yang diterapkan meliputi _rescaling_, _rotation_, _horizontal_ dan _vertical_,_ shearing_, _zooming_,_ width_shift_range_ dan , _height_shift_range_.              Setelah itu, Anda menggunakan flow_from_directory untuk membuat generator pelatihan dan validasi. kita menentukan _class_mode_ yaitu _'categorical_' untuk menghasilkan label kategori yang disandikan satu-panas. Gambar juga diubah ukurannya menjadi 150x150 piksel menggunakan parameter target_size.
 
## Modeling
membuat model cnn
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/843970a6-7f08-4b92-a15d-532505e96160)                    
Lapisan tambahan ditambahkan ke model. Lapisan _Flatten _digunakan untuk meratakan output, diikuti oleh lapisan _Dropout_ untuk mengurangi overfitting. Kemudian, dua lapisan yang terhubung sepenuhnya (Padat) dengan fungsi aktivasi rel ditambahkan, dan lapisan keluaran akhir dengan aktivasi sigmoid ditambahkan dengan 2 unit yang mewakili dua kelas (kucing dan anjing).
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/c3ba35be-d16d-4f4f-b3bf-9e820d160d19)
melakukan komplikasi model menggunakan fungsi optimizer adam yang digunakan untuk mengupdate iterasinya supaya lebih cepat mencapai titik yang lebih optimal
Model dilatih menggunakan fungsi fit
Data pelatihan disediakan oleh train_generator, dan data validasi disediakan oleh validasi_generator. Pelatihan dihentikan jika akurasi pelatihan dan validasi melebihi 0,80, seperti yang ditentukan dalam _callback_.

## Evaluation
Tujuan visualisasi ini adalah untuk membantu memahami bagaimana model berkembang selama pelatihan. Kita dapat mengamati apakah model cenderung _overfit_ atau _underfit_, serta mengkaji tren akurasi dan loss untuk setiap epoch. Visualisasi ini juga dapat membantu dalam pemilihan parameter dan pengambilan keputusan terkait dengan model.

![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/30c11a2b-fa09-4e3c-9057-3ae8f8bc1c12)
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/c5feb981-0fea-463e-9ee8-ba9fcac959c7)

Kode di atas digunakan untuk membuat plot yang menunjukkan perubahan akurasi dan loss model selama pelatihan.
Pertama, kami menggunakan plt.plot() untuk membuat plot garis untuk akurasi pelatihan (History.history['accuracy']) dan akurasi validasi (History.history['val_accuracy']).
Kemudian, plt.title() digunakan untuk memberikan judul plot sebagai "Akurasi Model".
Selanjutnya, plt.legend() digunakan untuk menampilkan legenda ("train" dan "test") di pojok kiri atas plot.
Terakhir, plt.show() digunakan untuk menampilkan plot akurasi.
Dengan menggunakan kode ini, kita dapat memvisualisasikan perubahan akurasi dan loss model selama pelatihan dengan plot yang disajikan. Plot ini membantu kami menganalisis dan memahami performa model secara visual.

**---Ini adalah bagian akhir laporan---**
