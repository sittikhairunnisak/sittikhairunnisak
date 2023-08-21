# Laporan Proyek Machine Learning - Sitti Khairunnisak
Machine learning Klasifikasi Gambar, menentukan gambar kucing atau anjing 

## Domain Proyek
Hewan merupakan makhluk hidup selalu ada di sekitar kita. Masyarakatpun banyak yang memelihara hewan peliharaan terutama anjing dan kucing, karena memiliki karakter dan fungsi yang beragam dan menyenangkan manusia. Pada pengolahan citra, proses pengklasifikasian objek merupakan salah satu bagian permasalahan dalam _computer Vision_. Tujuan pengklasifikasian citra ini adalah proses memasukkan citra kedalam beberapa kategori yang disesuaikan dengan kebutuhan. Ide dari pengklasifikasian citra yang spesifik dengan memberi masukkan dari sekumpulan angka yang diproses dan menghasilkan angka yang merupakan representasi dari kategori citra tersebut, dan hasil dari klasifikasi citra digital dapat menjadi alternatif dalam mengenali hewan. Selain itu proses mengklasifikasikan citra anjing dan kucing ini diharapkan adalah komputer dapat mengenali dan membedakan objek pada citra selayaknya manusia[1]

Pengenalan wajah menggunakan machine learning sangat penting. Dengan menggunakan machine learning, teknologi pengenalan wajah dapat mencapai tingkat akurasi yang tinggi dalam mengenali wajah hewan (kucing dan anjing) atau wajah seseorang. Algoritma machine learning seperti _neural network_ dapat meniru proses otak manusia dalam mengenali fitur-fitur khusus pada wajah, seperti jarak antara mata, tinggi dahi, lebar hidung, dan sebagainya. Algoritma _facial recognition_ dirancang untuk memetakan fitur wajah seseorang secara matematis.
selain itu Teknologi pengenalan wajah menggunakan machine learning dapat diterapkan dalam berbagai bidang, seperti keamanan, pengenalan identitas, sehingga dapat digunakan untuk mendeteksi ancaman dan memprediksi risiko keamanan.

## Business Understanding
Adapun dampak positifnya dari masalah pengenalan wajah adalah dapat memudahkan dalam memantau aktivitas anjing atau kucing secara _real-time_. Hal ini dapat membantu pemilik hewan peliharaan untuk memastikan bahwa hewan peliharaan aman dan tidak melakukan hal-hal yang tidak diinginkan, dan juga meningkatkan keamanan, jadi pemilik hewan peliharaan dapat memastikan bahwa hanya anjing atau kucing yang dapat masuk ke dalam rumah atau area tertentu. Hal ini dapat membantu mencegah anjing atau kucing yang tidak diinginkan masuk ke dalam rumah dan mengganggu hewan peliharaan yang ada di dalamnya. dan dengan penerapan teknologi pengenalan otomatis menggunakan machine learning, diharapkan dapat memberikan kemudahan, keamanan, dan kenyamanan bagi pemilik hewan peliharaan dalam merawat dan memantau aktivitas hewan peliharaan.

### Problem Statements
permasalahan yang ada dalam proyek pengenalan wajah kucing dan anjing adalah
- bagaimana proses klasifikasi dengan CNN mampu menghasilkan deteksi objek citra anjing dan kucing serta membedakan modelnya?
- Bagaimana penggunaan dataset training dengan resolusi citra (gambar) yang bagus dapat membuat model yang dipakai lebih baik dan mengurangi overfitting?
- Berapa % capaian akurasi dan presisi sistem klasifikasi terhadap penganalan anjing dan kucing ?

### Goals
penyelesaian mengenai masalah diatas adalah dengan
- langkah-langkah yaitu import beberapa libraries yang dapat mendeteksi gambar. Lalu mendefinisikan ukuran gambar yang ingin diterapkan. Selanjutnya menggunakan
dataset dalam mengkategorikan anjing dan kucing. Selanjutnya menggunakan dataset yang telah diperoleh untuk mendapatkan data yang dapat dilatih dari
total data yang didapatkan
- Menvisualisasikan grafik keakuratan pada proses training hingga validasi data yang diperoleh. Dan menggunakan Callback untuk mengurangi overftting
- pencapaian % akurasi dihentikan ketika 80%, akurasi tidak boleh turun di bawah 80% untuk mencegah model dari overfitting ke data pelatihan.

## Data Understanding
data yang digunakan mengimport dari kaggle, https://www.kaggle.com/datasets/tongpython/cat-and-dog
pertama Menentukan direktori, dari isi folder itu ada tiga yaitu bahan, latih dan validasi, lalu anda print jumlah data yang terdiri dari dua kategori, yaitu anjing dan kucing, akan muncul keterangan kategori kucing ada 1011 gambar dan untuk kategori anjing ada 1015 gambar. Setiap data dalam kumpulan data direpresentasikan dalam format file gambar seperti JPEG atau PNG. Setiap gambar memiliki ukuran dan resolusi yang berbeda. 

### Variabel-variabel pada cats and dog dataset adalah sebagai berikut:
Variabel dalam kumpulan data adalah gambar yang berupa kucing atau anjing. Gambar tersebut digunakan untuk melatih dan menguji algoritma pembelajaran mesin untuk mengklasifikasikan apakah suatu gambar berisi kucing atau anjing
- cats : merupakan hewan jenis kucing
- dogs : merupakan hewan jenis anjing

## Data Preparation
Data _Image generator_ digunakan untuk membuat gambar dari teks atau input data lainnya yang dapat membantu dalam pemrosesan data. _Image generator_ dapat membantu masalah pengolahan data, mampu membuat gambar tambahan dari yang sudah ada. Dengan menghasilkan gambar baru model dapat dilatih pada kumpulan data yang lebih besar dan beragam, yang dapat meningkatkan akurasinya. Langkah pertama adalah mengimpor pustaka yang diperlukan dengan cara import _imagedatagenerator._
proses pembagian data menggunakan _training_, _validation_ dan _testing_ dengan menulis rasio pembagian.
tensorflow untuk membuat dan melatih model.
_ImageDataGenerator _dari _tensorflow.keras.preprocessing.image _ untuk augmentasi data dan menyiapkan generator data untuk pelatihan dan validasi dan menggunakan _ImageDataGenerator _ untuk melakukan augmentasi data pada gambar pelatihan. Beberapa augmentasi yang diterapkan meliputi _rescaling_ dengan nilai 1/255, _rotation_range, 20 _horizontal_ dan _vertical_,_ shearing_, 0.2 _zooming_, 0.1 
_width_shift_range_ 0.2 dan , _height_shift_range_ 0.2. Setelah itu, menggunakan flow_from_directory untuk membuat generator pelatihan dan validasi. kita menentukan _class_mode_ yaitu _'categorical_'. Gambar juga diubah ukurannya menjadi 150x150 piksel menggunakan parameter target__size._
 
## Modeling
Membuat model cnn
Menggunakan layer konvolusi 2 dimensi pada model neural network yang digunakan untuk memproses data gambar dengan nilai (32, (3,3)) dan bentuk input gambar dengan ukuran 150x150 dengan 3 byte. Untuk layer konvolusi 2 dimensi yang kedua dengan nilai (64) dan aktivasi relu.
Lapisan tambahan ditambahkan ke model. Lapisan _Flatten _digunakan untuk meratakan output, diikuti oleh lapisan _Dropout_ untuk mengurangi overfitting. Kemudian, dua lapisan yang terhubung sepenuhnya (Padat) dengan fungsi aktivasi rel ditambahkan, dan lapisan keluaran akhir dengan aktivasi sigmoid ditambahkan dengan dua unit yang mewakili dua kelas (kucing dan anjing).
Melakukan komplikasi model menggunakan fungsi optimizer adam yang digunakan untuk mengupdate iterasinya supaya lebih cepat mencapai titik yang lebih optimal dan _Loss function_ Yang digunakan untuk Klasifikasi adalah _binary_.
Model dilatih menggunakan fungsi fit, dengan hasil train 1821 dan hasil validation 204 dari dua class dengan pembagian 90%:10% dan jumlah 
_epoch_ 40 dan setiap jumlah _step epochnya_ 40, karena itu adalah jumlah yang pas dari datanya dan untuk modelnya. Pelatihan dihentikan jika akurasi pelatihan dan validasi melebihi 0,80, seperti yang ditentukan dalam _callback_.

## Evaluation
Untuk membuat plot yang menunjukkan perubahan akurasi dan loss model selama pelatihan adalah:
Pertama, menggunakan plt.plot() untuk membuat plot garis untuk akurasi pelatihan, loss pelatihan dan akurasi, loss validasi.
Hasil penerapan pada metrik evaluasi adalah memberikan informasi tentang performa model, seperti kemampuan model dalam mengklasifikasikan data dengan benar, jenis kesalahan yang dibuat, dan tingkat kebenaran dari proses klasifikasi sehingga mendapatkan hasil model terbaik.
Kemudian, plt.title() digunakan untuk memberikan judul plot sebagai "Akurasi Model".
Selanjutnya, plt.legend() digunakan untuk menampilkan legenda di pojok kiri atau kanan atas plot.
Terakhir, plt.show() digunakan untuk menampilkan plot akurasi.
Plot ini membantu menganalisis dan memahami performa model secara visual. 
Hasil yang didapatkan untuk akurasi pelatihan adalah 0.80 dan akurasi validasi 0.76 dan untuk loss pelatihan 0.41 dan loss validasi 0.53
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/2e335c89-b4bc-4bcc-ae0a-c739f4b8adcb)
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/51ce4102-cfab-41c2-9373-858d7833043e)

Referensi: [1.] Suyanto, (2018), Machine Learning Tingkat Dasar dan Lanjut, Penerbit Informatika Bandung.

**---Ini adalah bagian akhir laporan---**
