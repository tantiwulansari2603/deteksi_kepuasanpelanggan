# import keras
# from keras.models import model_from_json

# json_file = open('emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)
# # model = keras.models.load_model('emotion_model.json')  # Gantilah dengan nama model Anda
# emotion_model.summary()

# import cv2
# from mtcnn import MTCNN

# # Inisialisasi MTCNN
# detector = MTCNN()

# # Inisialisasi objek VideoCapture untuk mengakses kamera
# cap = cv2.VideoCapture(0)  # Nomor 0 mengacu pada kamera utama, ganti dengan nomor kamera yang sesuai jika Anda memiliki lebih dari satu kamera.

# while True:
#     # Ambil frame dari video capture
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     # Deteksi wajah pada frame
#     faces = detector.detect_faces(frame)
    
#     # Gambar kotak di sekitar wajah yang terdeteksi
#     for face in faces:
#         x, y, width, height = face['box']
#         cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
    
#     # Tampilkan frame yang telah dimodifikasi dengan kotak wajah
#     cv2.imshow('Face Detection', frame)
    
#     # Tekan 'q' untuk keluar dari loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Bebaskan sumber daya dan tutup jendela tampilan
# cap.release()
# cv2.destroyAllWindows()

# import os
# import tensorflow as tf
# from keras.applications import VGG16

# # Tentukan path untuk berat model yang sudah diunduh
# local_weights_file = '/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

# # Pastikan berat model sudah ada di direktori lokal
# if not os.path.exists(local_weights_file):
#     raise Exception("File berat model VGG16 tidak ditemukan. Harap unduh manual dari sumbernya.")

# # Load pre-trained model VGG16 dari berat model lokal
# base_model = VGG16(weights=local_weights_file, include_top=False, input_shape=input_shape)

# Sisanya sama seperti kode sebelumnya

import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Device {i}: {cap.getBackendName()}")
        cap.release()

