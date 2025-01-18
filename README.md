# Laporan Proyek Machine Learning - Muhamad Amhar Rayadin

## Project Overview

Di era modern ini, teknologi informasi memainkan peran penting dalam berbagai aspek kehidupan, termasuk bidang kesehatan. Salah satu tantangan yang sering dihadapi oleh masyarakat adalah kesulitan dalam menentukan obat yang sesuai dengan gejala yang dirasakan tanpa harus berkonsultasi langsung dengan tenaga medis. Proses ini tidak hanya memakan waktu tetapi juga dapat meningkatkan risiko kesalahan dalam memilih obat yang dapat memperburuk kondisi pasien.

Sistem rekomendasi obat berbasis gejala hadir sebagai solusi untuk masalah ini. Sistem ini memanfaatkan algoritma pembelajaran mesin untuk menganalisis data gejala yang dimasukkan oleh pengguna dan memberikan rekomendasi obat yang sesuai. Dengan demikian, sistem ini dapat membantu masyarakat mendapatkan informasi yang lebih cepat, akurat, dan relevan terkait obat-obatan yang mereka butuhkan.  

Proyek ini menjadi penting untuk diselesaikan karena dapat meningkatkan aksesibilitas informasi medis, terutama bagi masyarakat di daerah terpencil yang sulit mendapatkan konsultasi langsung dari dokter. Selain itu, sistem ini dapat berkontribusi dalam mengurangi beban kerja tenaga medis dengan menyediakan rekomendasi awal yang dapat menjadi acuan sebelum konsultasi lebih lanjut.  

Sebuah studi yang dilakukan oleh [A. Sarker dan C. Gonzalez (2017)](https://doi.org/10.1016/j.jbi.2017.02.003) menunjukkan bahwa pemanfaatan teknologi berbasis data dalam analisis gejala dan pengobatan memiliki potensi besar dalam meningkatkan efisiensi layanan kesehatan. Penelitian lain oleh [M. Paul dan M. Dredze (2011)](https://dl.acm.org/doi/10.1145/2020408.2020422) menggarisbawahi bahwa data yang diolah dengan teknik machine learning dapat memberikan wawasan berharga untuk mendukung pengambilan keputusan di bidang kesehatan.

Dengan latar belakang ini, proyek sistem rekomendasi obat berdasarkan gejala yang dirasakan tidak hanya menawarkan solusi praktis tetapi juga berkontribusi dalam memajukan layanan kesehatan berbasis teknologi di Indonesia.

## Business Understanding  

Kesehatan adalah aspek penting dalam kehidupan manusia, dan akses terhadap informasi obat yang tepat menjadi kebutuhan mendesak, terutama bagi masyarakat yang mengalami gejala tertentu namun tidak memiliki akses langsung ke tenaga medis. Dalam konteks ini, sistem rekomendasi obat berbasis gejala dapat menjadi solusi untuk membantu masyarakat mendapatkan informasi awal mengenai obat yang sesuai dengan kondisi mereka.

Permasalahan utama yang dihadapi adalah bagaimana menciptakan sistem yang dapat memberikan rekomendasi obat secara relevan dan akurat berdasarkan deskripsi gejala pengguna. Selain itu, sistem ini harus mampu mengolah data secara efisien untuk memastikan rekomendasi yang diberikan tepat sasaran.

Proyek ini bertujuan untuk menjawab tantangan tersebut dengan mengembangkan sistem rekomendasi berbasis algoritma content-based filtering. Pendekatan ini memanfaatkan deskripsi gejala dan informasi obat dalam basis data untuk menghasilkan rekomendasi yang relevan. Dengan pendekatan ini, diharapkan sistem dapat memberikan manfaat nyata bagi pengguna, khususnya dalam meningkatkan aksesibilitas informasi obat yang cepat dan terpercaya. 

### Problem Statements  

Sistem rekomendasi obat berbasis gejala bertujuan untuk menjawab beberapa permasalahan berikut:  
1. Bagaimana mengolah data gejala dan informasi obat secara efisien untuk mendukung proses rekomendasi?
2. Bagaimana cara memberikan rekomendasi obat yang relevan berdasarkan gejala yang dirasakan oleh pengguna?  
3. Bagaimana memastikan rekomendasi obat yang diberikan akurat dan dapat diandalkan berdasarkan recall dan Mean Reciprocal Rank (MRR)?  

### Goals  

Proyek ini bertujuan untuk:  
1. Mengoptimalkan proses pengolahan data menggunakan algoritma yang efisien dan relevan.
2. Mengembangkan sistem yang mampu memberikan rekomendasi obat berdasarkan input gejala dari pengguna.  
3. Memastikan rekomendasi obat yang dihasilkan akurat dan sesuai dengan kebutuhan pengguna menggunakan metrik evaluasi recall dan Mean Reciprocal Rank (MRR). 

### Solution Approach  

Untuk mencapai tujuan di atas, digunakan pendekatan solusi sebagai berikut:  

1. **Data Processing dan Visualisasi**  
   - Dilakukan pembersihan dan analisis data untuk memastikan bahwa data yang digunakan relevan dan berkualitas.  
   - Visualisasi data membantu dalam memahami pola-pola tertentu dalam hubungan antara gejala dan obat.  

2. **Content-Based Filtering**  
   - Sistem rekomendasi ini menggunakan pendekatan *content-based filtering* dengan memanfaatkan algoritma TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer.  
   - Pendekatan ini menganalisis kemiripan antara deskripsi gejala yang dimasukkan pengguna dengan deskripsi obat yang ada di basis data.  
   - Dengan menghitung skor relevansi menggunakan TF-IDF, sistem dapat memberikan rekomendasi obat yang paling relevan dengan gejala pengguna.  

Pendekatan ini dipilih karena kemampuannya untuk menghasilkan rekomendasi yang spesifik dan relevan dengan preferensi pengguna, sekaligus memanfaatkan deskripsi data yang sudah ada tanpa memerlukan interaksi pengguna sebelumnya.

## Data Understanding  

Dataset yang digunakan dalam proyek ini berjudul **Medical Reccomandation dataSet**, yang berasal dari Kaggle Dataset dan dapat diakses melalui tautan berikut: [Medical Reccomandation dataSet](https://www.kaggle.com/datasets/joymarhew/medical-reccomadation-dataset). Dataset ini terdiri dari **287 baris data** dan **7 kolom**. Dataset ini berisi informasi mengenai gejala, penyebab, penyakit, dan obat-obatan yang relevan.

### Atribut Lengkap Dataset
Berikut adalah deskripsi lengkap dari setiap atribut dalam dataset:  

| Atribut      | Tipe Data     | Deskripsi                                                                 |  
|--------------|---------------|---------------------------------------------------------------------------|  
| `Name`         | String        | Nama pasien.                     |  
| `DateOfBirth`  | Date          | Tanggal lahir pasien.            |  
| `Gender`       | Categorical   | Jenis kelamin pasien (Male, Female).                                     |  
| `Symptoms`     | String        | Deskripsi gejala yang dirasakan pasien.                                  |  
| `Causes`       | String        | Penyebab gejala yang dirasakan pasien.                                   |  
| `Disease`      | String        | Penyakit yang mungkin terkait dengan gejala.                             |  
| `Medicine`     | String        | Obat atau tidakan yang direkomendasikan untuk mengatasi gejala atau penyakit terkait. |

### Missing Values dan Duplikasi Data
Dataset ini **memiliki beberapa nilai yang hilang** pada beberapa kolom, seperti yang ditunjukkan pada tabel berikut:  

| Kolom         | Missing Values |  
|---------------|----------------|  
| Name          | 46             |  
| DateOfBirth   | 46             |  
| Gender        | 45             |  
| Symptoms      | 40             |  
| Causes        | 42             |  
| Disease       | 38             |  
| Medicine      | 45             |  

10 baris pertama data null ditampilkan pada tabel berikut.
![Data Null](https://i.ibb.co.com/9gXxnv1/data-null.png) 
Berdasarkan tabel, data null tidak dapat digunakan karena sebagian besar atributnya null sehingga tidak ada insight yang diperoleh.

Selain itu, **terdapat 84 baris data duplikat** dalam dataset. Contoh data duplikat ditampilkan pada tabel berikut.
![Data Duplikat](https://i.ibb.co.com/b2xNKkH/data-duplikat.png)
Berdasarkan tabel, data duplikat tidak sama persis di setiap atributnya dan terdapat kesamaan pada beberapa atribut saja, misal `Name` dan `DateOfBirth`. Jika dianalisis, adalah hal yang wajar jika seseorang mengalami beberapa kali riwayat gejala sakit dan obat yang diminumnya sehingga data seperti ini tetap dipertahankan.

### Broken Data
Terdapat beberapa data yang invalid, seperti adanya pemenggalan kalimat yang tidak sesuai, seperti yang ditunjukkan pada tabel berikut.
![Broken Data](https://i.ibb.co.com/rdMh4WH/broken-data.png)
Olehnya itu terlebih dahulu dilakukan perbaikan untuk data seperti ini agar kualias data yang digunakan lebih baik.

### Univariate EDA
Dilakukan tahapan untuk analisis tiap atribut secara individu, menampilkan top 10 data terbanyak tiap atribut yang akan digunakan seperti berikut:
![Top data](https://i.ibb.co.com/y8mxyCP/top10da.png)



## Data Preparation

Pada tahap ini dilakukan persiapan data sebelum masuk ke pemodelan. Adapun tahapan yang dilakukan adalah sebagai berikut.

1. Memperbaiki broken data. Hal ini dilakukan dengan filter data yang tidak sesuai seperti kesalahan pemenggalan kata kemudian menggantinya dengan kata yang sesuai.
2. Menghapus null data. Hal ini dilakukan karena data null yang ada tidak memungkinkan untuk diperbaiki karena hampir setiap kolomnya berisi null value.
3. Merubah format data menjadi huruf kecil (lower case). Hal ini dilakukan agar data lebih mudah diproses oleh model dan agar model tidak case sensitive.
4. Memilih fitur yang digunakan dalam model dengan menggabungkan beberapa atribut yaitu `Symptoms`, `Causes`, dan `Disease` ke fitur baru bernama `combined_features` yang menjadi dasar analisis. Atribut yang digunakan sebagai target adalah `Medicine`. Adapun fitur lain tidak digunakan karena tidak relevan dengan metode yang akan digunakan yaitu content based filtering. 
5. Menghilangkan karakter yang tidak diperlukan pada atribut fitur seperti dan koma (,) agar lebih muda diproses oleh model.
6. Menggunakan TF-IDF Vectorization. Algortima ini digunakan untuk merepresentasikan `combined_features` sebagai vektor numerik. Setiap kata dalam teks diberi bobot berdasarkan *Term Frequency (TF)* (Seberapa sering kata tersebut muncul dalam fitur gabungan) dan *Inverse Document Frequency (IDF)* (Seberapa unik kata tersebut di seluruh dataset). Hasilnya adalah matriks TF-IDF, di mana setiap baris merepresentasikan sebuah obat, dan setiap kolom adalah bobot kata yang relevan.


## Modeling  

Sistem rekomendasi yang dibangun menggunakan pendekatan *item-based filtering* dengan menggunakan cosine similarity untuk menganalisis kemiripan antar obat berdasarkan vector tf-idf dari fitur gabungan (*combined features*). 
Pendekatan *item-based filtering* digunakan untuk menganalisis kemiripan antar item (dalam hal ini obat) dengan *cosine similarity* berdasarkan fitur deskriptif. Penggunaan teknik ini memungkinkan sistem untuk memahami hubungan antar obat dengan cara berikut:    

1. **Penghitungan Kemiripan**:  
   Setelah fitur teks diubah menjadi matriks TF-IDF, metrik *cosine similarity* digunakan untuk menghitung kemiripan antar obat. Hasilnya adalah matriks kemiripan (*similarity matrix*) yang menunjukkan tingkat hubungan antara setiap pasangan obat.  

2. **Rekomendasi Berdasarkan Kemiripan Obat**:  
   - Untuk setiap obat yang dicari, sistem memeriksa baris dalam matriks kemiripan yang sesuai dengan obat tersebut.  
   - Obat-obatan dengan skor kemiripan tertinggi dipilih sebagai rekomendasi.  

3. **Rekomendasi Berdasarkan Gejala yang Dirasakan**
   - Mencari obat yang sesuai berdasarkan gejala yang dirasakan dan menjadikannya reference medicine.
   - Reference medicine digunakan untuk menghasilkan top-n rekomendasi berdasarkan kemiripan obat tersebut menggunakan matriks kemiripan yang dibuat sebelumnya.

### Kelebihan dan Kekurangan  

**Kelebihan:**  
- **Spesifik dan relevan**: Pendekatan ini menghasilkan rekomendasi yang sangat relevan berdasarkan kemiripan antar obat.  
- **Efisien untuk dataset berbasis teks**: TF-IDF bekerja baik dalam memahami teks deskriptif seperti gejala dan obat.  

**Kekurangan:**  
- **Cold start problem untuk item**: Obat baru tanpa data fitur tidak dapat dianalisis.  
- **Skalabilitas terbatas**: Perhitungan matriks kemiripan bisa menjadi lambat untuk dataset yang sangat besar.  

### Output  
Sistem ini menghasilkan *top-3 recommendations* berupa obat-obatan yang relevan berdasarkan gejala atau obat referensi yang diberikan pengguna.  

```python
recommendations = get_recommendations_by_symptoms('fatigue')
print(recommendations)
```
output:
```
Medicine
sumatriptan           1.000000
beta-blockers         0.631797
relaxation, nsaids    0.261742
Name: sumatriptan, dtype: float64
```

## Evaluation  

Tahap evaluasi bertujuan untuk mengukur kinerja sistem rekomendasi yang dikembangkan. Evaluasi dilakukan dengan menggunakan dua metrik utama, yaitu **Recall@k** dan **Mean Reciprocal Rank (MRR)**.  

### Metrik Evaluasi  

1. **Recall@k**  
   Recall@k adalah metrik yang mengukur sejauh mana sistem mampu merekomendasikan obat yang benar (sesuai dengan data referensi) di antara *top-k recommendations*.  

   - **Persamaan**:  
     \[
     Recall@k = \frac{\text{Jumlah obat yang relevan dalam rekomendasi}}{\text{Jumlah obat yang relevan sebenarnya}}
     \]
     Karena setiap query hanya memiliki satu obat benar (true medicine), recall dihitung sebagai:  
     \[
     Recall@k = \frac{|R \cap T|}{|T|}
     \]
     Di mana \(R\) adalah set obat yang direkomendasikan, dan \(T\) adalah obat yang benar.  

   - **Cara Kerja**:  
     - Sistem menghasilkan *top-k recommendations* untuk setiap query.  
     - Dihitung apakah obat yang benar termasuk dalam rekomendasi tersebut.  

2. **Mean Reciprocal Rank (MRR)**  
   MRR adalah metrik yang mengukur rata-rata kebalikan dari peringkat (*rank*) di mana rekomendasi yang benar pertama kali muncul.  

   - **Persamaan**:  
     \[
     MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{rank_i}
     \]
     Di mana \(N\) adalah jumlah query, dan \(rank_i\) adalah posisi obat yang benar pada rekomendasi ke-\(i\).  

   - **Cara Kerja**:  
     - Sistem mencatat posisi (*rank*) dari obat yang benar dalam *top-k recommendations*.  
     - Reciprocal rank dihitung sebagai kebalikan dari posisi tersebut.  

### Proses Evaluasi  

1. **Pengambilan Sampel Data Uji**  
   Dataset dibagi menjadi data latih dan data uji. Dalam kode, sampel data uji diambil secara acak menggunakan `data.sample()`.  

2. **Menghitung Recall@k dan MRR**  
   - Untuk setiap query pada data uji, sistem menghasilkan *top-k recommendations*.  
   - Recall@k dihitung berdasarkan keberadaan obat yang benar dalam rekomendasi.  
   - MRR dihitung berdasarkan posisi obat yang benar dalam rekomendasi.  

3. **Rata-rata Skor**  
   - Recall@k dan MRR dihitung rata-ratanya untuk semua query dalam data uji.  


### Hasil Evaluasi  

Setelah menjalankan proses evaluasi, berikut adalah hasil yang diperoleh:  
```plaintext
Evaluation Results: {'Recall@k': 0.87, 'MRR': 0.73}
```

- **Recall@k = 0.87**: Sistem berhasil merekomendasikan obat yang benar dalam 87% query.  
- **MRR = 0.73**: Obat yang benar rata-rata muncul di peringkat ke-1 hingga ke-2 dalam rekomendasi.  

Hasil evaluasi menunjukkan bahwa sistem rekomendasi memiliki kinerja yang baik dalam memberikan rekomendasi obat yang relevan dan akurat. Namun, metrik ini dapat ditingkatkan lebih lanjut dengan optimasi algoritma atau penanganan data yang lebih baik.  


## Penutup  

Proyek sistem rekomendasi obat berbasis gejala ini telah berhasil dikembangkan dengan pendekatan *content-based filtering* menggunakan algoritma TF-IDF Vectorizer. Berdasarkan hasil evaluasi menggunakan metrik Recall@k dan Mean Reciprocal Rank (MRR), sistem menunjukkan performa yang cukup baik, dengan nilai Recall@k sebesar 0.87 dan MRR sebesar 0.73. Hal ini menunjukkan bahwa sistem mampu merekomendasikan obat yang relevan secara konsisten dan menempatkan obat yang benar pada posisi yang tinggi dalam daftar rekomendasi.  

Sistem ini telah memenuhi tujuan awal proyek, yaitu:  
1. Memberikan rekomendasi obat berdasarkan input gejala yang dirasakan pengguna.  
2. Memastikan rekomendasi obat yang diberikan relevan dan akurat.  
3. Menggunakan pendekatan algoritma yang efisien untuk pengolahan data gejala dan obat.  

Namun, ada beberapa tantangan yang dapat menjadi fokus pengembangan di masa depan, seperti:  
- Mengatasi masalah *cold start* untuk obat-obatan baru yang belum memiliki deskripsi lengkap.  
- Meningkatkan skalabilitas sistem agar dapat menangani dataset yang lebih besar dengan efisiensi tinggi.  
- Menyempurnakan kualitas data untuk menghasilkan rekomendasi yang lebih akurat.  

Proyek ini menunjukkan potensi besar dalam membantu masyarakat mengakses informasi obat dengan cepat dan efisien, terutama bagi mereka yang memiliki keterbatasan akses ke layanan medis. Dengan pengembangan lebih lanjut, sistem ini dapat menjadi solusi yang lebih komprehensif dalam mendukung layanan kesehatan berbasis teknologi.  