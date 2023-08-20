# Laporan Proyek Machine Learning - Sitti Khairunnisak
Machine learning Klasifikasi Gambar, menentukan gambar kucing atau anjing

## Domain Proyek
Hewan merupakan makhluk hidup selalu ada di sekitar kita. Masyarakatpun banyak yang memelihara hewan peliharaan terutama anjing dan kucing, karena memiliki karakter dan fungsi yang beragam dan menyenangkan manusia. Pada pengolahan citra, proses pengklasifikasian objek merupakan salah satu bagian permasalahan dalam computer Vision. Tujuan pengklasifikasian citra ini adalah proses memasukkan citra kedalam beberapa kategori yang disesuaikan dengan kebutuhan. Ide dari pengklasifikasian citra yang spesifik dengan memberi masukkan dari sekumpulan angka yang diproses dan menghasilkan angka yang merupakan representasi dari kategori citra tersebut, dan hasil dari klasifikasi citra digital dapat menjadi alternatif dalam mengenali hewan. Selain itu proses mengklasifikasikan citra anjing dan kucing ini diharapkan adalah komputer dapat mengenali dan membedakan objek pada citra selayaknya manusia.
Pengenalan wajah menggunakan machine learning sangat penting. Dengan menggunakan machine learning, teknologi pengenalan wajah dapat mencapai tingkat akurasi yang tinggi dalam mengenali wajah hewan (kucing dan anjing) atau wajah seseorang. Algoritma machine learning seperti neural network dapat meniru proses otak manusia dalam mengenali fitur-fitur khusus pada wajah, seperti jarak antara mata, tinggi dahi, lebar hidung, dan sebagainya. Algoritma facial recognition dirancang untuk memetakan fitur wajah seseorang secara matematis.
selain itu Teknologi pengenalan wajah menggunakan machine learning dapat diterapkan dalam berbagai bidang, seperti keamanan, pengenalan identitas, sehingga dapat digunakan untuk mendeteksi ancaman dan memprediksi risiko keamanan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Penyelesaian menentukan citra atau objen kucing dan anjing tersebut dengan cara membuat program dengan cnn yang dapat mengenali bentuk hewan kucing dan anjing.
- Penambahan dataset yang digunakan untuk training dengan resolusi gambar yang bagus dapat membuat model yang dipakai lebih baik dan mengurangi overfitting
  Format Referensi:  
  (Riyadi, A. S., Wardhani, I. P., & Widayati, S. (2021, September). Klasifikasi citra anjing dan kucing menggunakan metode convolutional neural network (CNN). In Prosiding Seminar SeNTIK (Vol. 5, No. 1, pp. 307-311).) 

## Business Understanding
apa dampak positif dari pemecahan masalah pengenalan wajah hewan pada  pemilik hewan peliharaan?
dampak positifnya adalah adalah dapat memudahkan dalam memantau aktivitas anjing atau kucing mereka secara real-time. Hal ini dapat membantu pemilik hewan peliharaan untuk memastikan bahwa hewan peliharaan mereka aman dan tidak melakukan hal-hal yang tidak diinginkan, dan juga meningkatkan keamanan, jadi pemilik hewan peliharaan dapat memastikan bahwa hanya anjing atau kucing mereka yang dapat masuk ke dalam rumah atau area tertentu. Hal ini dapat membantu mencegah anjing atau kucing yang tidak diinginkan masuk ke dalam rumah dan mengganggu hewan peliharaan yang ada di dalamnya. dan dengan penerapan teknologi pengenalan otomatis menggunakan machine learning, diharapkan dapat memberikan kemudahan, keamanan, dan kenyamanan bagi pemilik hewan peliharaan dalam merawat dan memantau aktivitas hewan peliharaan mereka.

### Problem Statements
- bagaimana proses klasifikasi dengan CNN mampu menghasilkan deteksi objek citra anjing dan kucing serta membedakan modelnya?
- Bagaimana penggunaan dataset training dengan resolusi citra (gambar) yang bagus dapat membuat model yang dipakai lebih baik dan mengurangi overfitting?
- Berapa % capaian akurasi dan presisi sistem klasifikasi terhadap penganalan anjing dan kucing ?

### Goals
- langkah-langkah yaitu import beberapa libraries yang dapat mendeteksi gambar. Lalu mendefinisikan ukuran gambar yang ingin diterapkan. Selanjutnya menggunakan
dataset dalam mengkategorikan anjing dan kucing. Selanjutnya penulis menggunakan dataset yang telah diperoleh untuk mendapatkan data yang dapat dilatih dari
total data yang didapatkan
- Menvisualisasikan grafik keakuratan pada proses training hingga validasi data yang diperoleh. Dan menggunakan Callback untuk mengurangi overftting
- % capaian akurasi berhenti ketika 80 sesuai target callback

## Data Understanding
data yang digunakan mengimport dari kaggle, https://www.kaggle.com/datasets/tongpython/cat-and-dog
dataset terdiri dari dua kategori, yaitu anjing dan kucing. kategori kucing ada 1011 gambar dan untuk kategori anjing ada 1015 gambar
print('Jumlah total gambar cats:', len(os.listdir(cats_dir)))                   
print('Jumlah total gambar dogs:', len(os.listdir(dogs_dir)))                     
Jumlah total gambar cats: 1011
Jumlah total gambar dogs: 1015
Dataset terdiri dari sejumlah gambar kucing dan anjing. Jumlah pasti gambar dalam kumpulan data dapat bervariasi dan bergantung pada ukuran kumpulan data yang digunakan. 
Setiap data dalam kumpulan data direpresentasikan dalam format file gambar seperti JPEG atau PNG. Setiap gambar memiliki ukuran dan resolusi yang berbeda.
contoh visualisasi kucing  alam dataset pengenalan otomatis anjing dan kucing menggunakan teknologi machine learning yang spesifik.
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/70e72744-0d42-4959-9393-489768df685d)

### Variabel-variabel pada cats and dog dataset adalah sebagai berikut:
- cats : merupakan hewan jenis kucing
- dogs : merupakan hewan jenis anjing

## Data Preparation
Mengapa menggunakan image generator?
Karena Image generator digunakan untuk membuat gambar dari teks atau input data lainnya yang dapat membantu dalam pemrosesan data. Image generator dapat membantu masalah pengolahan data, mampu membuat membuat gambar tambahan dari yang sudah ada. Dengan menghasilkan gambar baru model dapat dilatih pada kumpulan data yang lebih besar dan beragam, yang dapat meningkatkan akurasinya. Langkah pertama adalah mengimpor pustaka yang diperlukan dengan cara import imagedatagenerator seperti contoh kode ini, 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tensorflow untuk membuat dan melatih model.
ImageDataGenerator dari tensorflow.keras.preprocessing.image untuk augmentasi data dan menyiapkan generator data untuk pelatihan dan validasi.
dan menggunakan ImageDataGenerator untuk melakukan augmentasi data pada gambar pelatihan. Beberapa augmentasi yang diterapkan meliputi rescaling, rotation, horizontal dan vertical shift, shearing, zooming, dan horizontal flipping.
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,                                                        
    horizontal_flip = True,                                                     
    shear_range = 0.2,                                                          
    fill_mode = 'nearest',                                                       
    width_shift_range = 0.2,                                                   
    height_shift_range = 0.2,                                                    
    zoom_range = 0.1,                                                       
Setelah itu, Anda menggunakan flow_from_directory untuk membuat generator pelatihan dan validasi. kita menentukan class_mode='categorical' untuk menghasilkan label kategori yang disandikan satu-panas. Gambar juga diubah ukurannya menjadi 150x150 piksel menggunakan parameter target_size.
train_generator = train_datagen.flow_from_directory(                              
    train_dir,                                                                    
    target_size=(150,150),                                                      
    batch_size = 32,
    class_mode ='categorical'
 )
 
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
tf.keras.layers.Dense(2, activation= 'sigmoid')                                   
Lapisan tambahan ditambahkan ke model. Lapisan Flatten digunakan untuk meratakan output, diikuti oleh lapisan Dropout untuk mengurangi overfitting. Kemudian, dua lapisan yang terhubung sepenuhnya (Padat) dengan fungsi aktivasi rel ditambahkan, dan lapisan keluaran akhir dengan aktivasi sigmoid ditambahkan dengan 2 unit yang mewakili dua kelas (kucing dan anjing).

membentuk model summary
dan melakukan komplikasi model menggunakan fungsi optimizer adam yang digunakan untuk mengupdate iterasinya supaya lebih cepat mencapai titik yang lebih optimal
model.compile(loss='binary_crossentropy',                                        
              optimizer=tf.optimizers.Adam(),                                     
              metrics=['accuracy'])  
              
    history= model.fit(
    train_generator,                                                              
    steps_per_epoch = 40,                                                         
    epochs = 40,                                                                  
    validation_data = validation_generator,                                       
    validation_steps = 2,                                                        
    verbose =2,
    callbacks=[callbacks])
Model dilatih menggunakan fungsi fit
Data pelatihan disediakan oleh train_generator, dan data validasi disediakan oleh validasi_generator. Pelatihan dihentikan jika akurasi pelatihan dan validasi melebihi 0,80, seperti yang ditentukan dalam callback.

## Evaluation
Tujuan visualisasi ini adalah untuk membantu memahami bagaimana model berkembang selama pelatihan. Kita dapat mengamati apakah model cenderung overfit atau underfit, serta mengkaji tren akurasi dan loss untuk setiap epoch. Visualisasi ini juga dapat membantu dalam pemilihan parameter dan pengambilan keputusan terkait dengan model.

![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/30c11a2b-fa09-4e3c-9057-3ae8f8bc1c12)
![image](https://github.com/sittikhairunnisak/sittikhairunnisak/assets/132251307/c5feb981-0fea-463e-9ee8-ba9fcac959c7)

Kode di atas digunakan untuk membuat plot yang menunjukkan perubahan akurasi dan loss model selama pelatihan.
Pertama, kami menggunakan plt.plot() untuk membuat plot garis untuk akurasi pelatihan (History.history['accuracy']) dan akurasi validasi (History.history['val_accuracy']).
Kemudian, plt.title() digunakan untuk memberikan judul plot sebagai "Akurasi Model".
Selanjutnya, plt.legend() digunakan untuk menampilkan legenda ("train" dan "test") di pojok kiri atas plot.
Terakhir, plt.show() digunakan untuk menampilkan plot akurasi.
Dengan menggunakan kode ini, kita dapat memvisualisasikan perubahan akurasi dan loss model selama pelatihan dengan plot yang disajikan. Plot ini membantu kami menganalisis dan memahami performa model secara visual.

**---Ini adalah bagian akhir laporan---**
