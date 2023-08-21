# Laporan Proyek Machine Learning - Sitti Khairunnisak
Machine learning Klasifikasi Gambar, menentukan gambar kucing atau anjing 

## Domain Proyek
Hewan merupakan makhluk hidup yang banyak masyarakat memelihara, hewan peliharaan terutama anjing dan kucing, karena memiliki karakter dan fungsi yang beragam dan menyenangkan manusia. Pada pengolahan citra, proses pengklasifikasian objek merupakan salah satu bagian permasalahan dalam _computer Vision_. Tujuan pengklasifikasian citra ini adalah proses memasukkan citra kedalam beberapa kategori yang disesuaikan dengan kebutuhan. Ide dari pengklasifikasian citra yang spesifik dengan memberi masukkan dari sekumpulan angka yang diproses dan menghasilkan angka yang merupakan representasi dari kategori citra tersebut, dan hasil dari klasifikasi citra digital dapat menjadi alternatif dalam mengenali hewan. Selain itu proses mengklasifikasikan citra anjing dan kucing ini diharapkan adalah komputer dapat mengenali dan membedakan objek pada citra selayaknya manusia[1]

Pengenalan wajah menggunakan _machine learning _ sangat penting. Dengan menggunakan _machine learning_, teknologi pengenalan wajah dapat mencapai tingkat akurasi yang tinggi dalam mengenali wajah hewan (kucing dan anjing) atau wajah seseorang. Algoritme _machine learning _ seperti _neural network_ dapat meniru proses otak manusia dalam mengenali fitur-fitur khusus pada wajah, seperti jarak antara mata, tinggi dahi, lebar hidung, dan sebagainya. Algoritme _facial recognition_ dirancang untuk memetakan fitur wajah seseorang secara matematis.
Selain itu Teknologi pengenalan wajah menggunakan _machine learning_ dapat diterapkan dalam berbagai bidang, seperti keamanan, pengenalan identitas, sehingga dapat digunakan untuk mendeteksi ancaman dan memprediksi risiko keamanan.

## Business Understanding
Adapun dampak positifnya dari masalah pengenalan wajah adalah dapat memudahkan dalam memantau aktivitas anjing atau kucing secara _real-time_. Hal ini dapat membantu pemilik hewan peliharaan untuk memastikan bahwa hewan peliharaan aman dan tidak melakukan hal-hal yang tidak diinginkan, dan juga meningkatkan keamanan, jadi pemilik hewan peliharaan dapat memastikan bahwa hanya anjing atau kucing yang dapat masuk ke dalam rumah atau area tertentu. Hal ini dapat membantu mencegah anjing atau kucing yang tidak diinginkan masuk ke dalam rumah dan mengganggu hewan peliharaan yang ada di dalamnya. Dengan penerapan teknologi pengenalan otomatis menggunakan _machine learning_, diharapkan dapat memberikan kemudahan, keamanan, dan kenyamanan bagi pemilik hewan peliharaan dalam merawat dan memantau aktivitas hewan peliharaan.

### Problem Statements
Permasalahan yang ada dalam proyek pengenalan wajah kucing dan anjing adalah
- bagaimana proses klasifikasi dengan _CNN_ mampu menghasilkan deteksi objek citra anjing dan kucing serta membedakan modelnya?
- Bagaimana penggunaan dataset training dengan resolusi citra (gambar) yang bagus dapat membuat model yang dipakai lebih baik dan mengurangi _overfitting_?
- Berapa % capaian akurasi dan presisi sistem klasifikasi terhadap penganalan anjing dan kucing ?

### Goals
Penyelesaian mengenai masalah diatas adalah dengan
- langkah-langkah yaitu import beberapa _libraries_ yang dapat mendeteksi gambar. Lalu mendefinisikan ukuran gambar yang ingin diterapkan. Selanjutnya menggunakan dataset yang telah diperoleh untuk mendapatkan data yang dapat dilatih dari
total data yang didapatkan
- _Menvisualisasikan_ grafik keakuratan pada proses _training _hingga validasi data yang diperoleh. Dan menggunakan _Callback_ untuk mengurangi _overftting_
- pencapaian % akurasi dihentikan ketika 80%, akurasi tidak boleh turun di bawah 80% untuk mencegah model dari _overfitting_ ke data pelatihan.

## Data Understanding
Data yang digunakan _mengimport_ dari _kaggle_ , https://www.kaggle.com/datasets/tongpython/cat-and-dog
pertama Menentukan direktori, dari isi folder itu ada tiga yaitu bahan, latih dan validasi, lalu anda _print_ jumlah data yang terdiri dari dua kategori, yaitu anjing dan kucing, akan muncul keterangan kategori kucing ada 1011 gambar dan untuk kategori anjing ada 1015 gambar. Setiap data dalam kumpulan data _direpresentasikan_ dalam format file gambar seperti JPEG atau PNG. Setiap gambar memiliki ukuran dan resolusi yang berbeda. 

### Variabel-variabel pada cats and dog dataset adalah sebagai berikut:
Variabel dalam kumpulan data adalah gambar yang berupa kucing atau anjing. Gambar tersebut digunakan untuk melatih dan menguji algoritma pembelajaran mesin untuk mengklasifikasikan apakah suatu gambar berisi kucing atau anjing
- cats : merupakan hewan jenis kucing
- dogs : merupakan hewan jenis anjing

## Data Preparation
Data _Image generator_ digunakan untuk membuat gambar dari teks atau input data lainnya yang dapat membantu dalam pemrosesan data. _Image generator_ dapat membantu masalah pengolahan data, mampu membuat gambar tambahan dari yang sudah ada. Dengan menghasilkan gambar baru model dapat dilatih pada kumpulan data yang lebih besar dan beragam, yang dapat meningkatkan akurasinya. Langkah pertama adalah mengimpor pustaka yang diperlukan dengan cara _import_ _imagedatagenerator._
__Training set_ digunakan untuk melatih model dan mengoptimalkan parameter, sedangkan _testing set_ digunakan untuk menguji performa model yang telah dilatih pada data yang belum pernah dilihat sebelumnya. Rasio pembagian _dataset_ antara _training set_ dan _testing set_ adalah (90%:10%) menghasilkan _train_ 1821 dan hasil _validation_ 204 dari dua kelas.
_tensorflow_ untuk membuat dan melatih model.
_ImageDataGenerator_ dari _tensorflow.keras.preprocessing.image_ untuk augmentasi data dan menyiapkan generator data untuk pelatihan dan validasi dan menggunakan _ImageDataGenerator_ untuk melakukan augmentasi data pada gambar pelatihan. Beberapa augmentasi yang diterapkan meliputi _rescaling_ dengan nilai 1/255, _rotation_range_, 20 _horizontal_ dan _vertical_shearing_, 0.2 _zooming_, 0.1 
_width_shift_range_ 0.2 dan , _height_shift_range_ 0.2. Setelah itu, menggunakan flow_from_directory untuk membuat generator pelatihan dan validasi. kita menentukan _class_mode_ yaitu _'categorical_'. Gambar juga diubah ukurannya menjadi 150x150 piksel menggunakan parameter target _size._
 
## Modeling
Membuat model _cnn_ dengan jumlah _hidden_ layer tiga, terdapat tiga layer _Conv2D_ dan tiga layer _MaxPooling2D_.
Menggunakan layer konvolusi 2 dimensi pada model _neural network_ yang digunakan untuk memproses data gambar dengan nilai (32, (3,3)) dan bentuk input gambar dengan ukuran 150x150 dengan 3 _byte_. Untuk layer konvolusi 2 dimensi yang kedua dengan nilai (64) dan aktivasi relu.
Lapisan tambahan ditambahkan ke model. Lapisan _Flatten_ digunakan untuk meratakan _output_, diikuti oleh lapisan _Dropout_ untuk mengurangi _overfitting_. Kemudian, dua lapisan yang terhubung sepenuhnya (Padat) dengan fungsi aktivasi rel ditambahkan, dan lapisan keluaran akhir dengan aktivasi _sigmoid_ ditambahkan dengan dua unit yang mewakili dua kelas (kucing dan anjing).
Melakukan komplikasi model menggunakan fungsi _optimizer adam_ yang digunakan untuk _mengupdate iterasinya_ supaya lebih cepat mencapai titik yang lebih optimal dan _Loss function_ Yang digunakan untuk Klasifikasi adalah _binary_.
Model dilatih menggunakan fungsi fit, dengan jumlah _epoch_ 40 dan setiap jumlah _step epochnya_ 40, karena itu adalah jumlah yang pas dari datanya dan untuk modelnya. Pelatihan dihentikan jika akurasi pelatihan dan validasi melebihi 0,80, seperti yang ditentukan dalam _callback_.

## Evaluation
Untuk membuat plot yang menunjukkan perubahan akurasi dan _loss_ model selama pelatihan adalah:
Pertama, menggunakan _plt.plot()_ untuk membuat plot garis untuk akurasi pelatihan, _loss_ pelatihan dan akurasi, _loss_ validasi.
Hasil penerapan pada metrik evaluasi adalah memberikan informasi tentang performa model, seperti kemampuan model dalam mengklasifikasikan data dengan benar, jenis kesalahan yang dibuat, dan tingkat kebenaran dari proses klasifikasi sehingga mendapatkan hasil model terbaik.
Kemudian, _plt.title()_ digunakan untuk memberikan judul plot sebagai "Akurasi Model".
Selanjutnya, _plt.legend()_ digunakan untuk menampilkan legenda di pojok kiri atau kanan atas plot.
Terakhir, _plt.show()_ digunakan untuk menampilkan plot akurasi.
Plot ini membantu menganalisis dan memahami performa model secara visual. 
Hasil yang didapatkan untuk akurasi pelatihan adalah 0.80 dan akurasi validasi 0.76, ini adalah hasil yang bagus karena Akurasi pelatihan sebesar 0.80 dan akurasi validasi sebesar 0.76 menunjukkan bahwa model dapat memprediksi dengan benar sekitar 80% data pelatihan dan 76% data validasi. Semakin tinggi akurasi, semakin baik performa model.
Untuk loss pelatihan 0.41 dan loss validasi 0.53 nilai ini adalah hasil yang bagus, karena hasil ini menunjukkan bahwa model dapat meminimalkan kesalahan dalam memprediksi data. Semakin rendah loss, semakin baik performa model.

Referensi: [1.] Suyanto, (2018), Machine Learning Tingkat Dasar dan Lanjut, Penerbit Informatika Bandung. 

**---Ini adalah bagian akhir laporan---**
