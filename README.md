# bangkit-caps-smartgate
Belum ada waktu untuk update kode sesuai dengan aturan python dan library yang terbaru, jadi untuk sementara bisa berjalan dengan environmen khusus.
  ## Requirements
  Python        == 3.6.8
  
  argparse      == 1.4.0
  
  imutils       == 0.5.3
  
  keras         == 2.4.3
  
  matplotlib    == 3.2.2
  
  numpy         == 1.19.5
  
  opencv-python == 4.3.0.38
  
  pillow        == 7.2.0
  
  scipy         == 1.4.1
  
  scikit-learn  == 0.23.1
  
  tensorflow    == 2.2.0
  
  streamlit     == 0.79.0
  
  ## Langkah - Langkah
  Persyaratan libary dapat diinstall dengan menggunakan requirement.txt yang sudah disediakan.
  ```
  pip install -r requirement.txt
  ```
  Untuk membuat model dengan dataset baru dapat dipanggil dengan cara:
  ```
  python train_model.py --dataset <nama folder dataset>
  ```
  
  setelah model dibuat dengan nama file mask_detector.model, Kita bisa langsung mempraktikkan deteksi masker.
  ```
  python deteksi_masker_lewatVId.py
  ```
  Keluar dari simulasi cukup menekan tombol 'Q' pada keyboard.

  ## Overview
  Plot hasil dari training yang telah kami lakukan\
  ![image](https://user-images.githubusercontent.com/58261801/121347926-eebf9900-c951-11eb-8a41-3184cd0966bb.png)
