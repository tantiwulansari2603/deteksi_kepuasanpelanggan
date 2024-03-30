# from flask import Flask, render_template, Response, jsonify, request
# from flask_socketio import SocketIO, emit
# import cv2
# import time
# import numpy as np
# from keras.models import model_from_json
# # from keras.models import load_model
# from collections import Counter
# import json
# import os
# from keras import backend as K

# # import eventlet

# app = Flask(__name__)
# socketio = SocketIO(app)

# json_file = open('emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# # Load model klasifikasi ekspresi wajah (gantilah dengan jalur yang sesuai)
# emotion_model.load_weights("emotion_model.h5")

# # # Load detektor wajah (misalnya, menggunakan OpenCV Haar Cascades)
# face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# duration = 120
# image_interval = 2

# expression_results = []
# puas_points = 0
# tidak_puas_points = 0

# # Direktori untuk menyimpan gambar
# image_dir = "expression_images"

# # Fungsi untuk membuat folder jika belum ada
# def create_image_folder():
#     if not os.path.exists(image_dir):
#         os.makedirs(image_dir)

# # Panggil fungsi untuk membuat folder
# create_image_folder()

# # Baca poin ekspresi dari file jika ada
# try:
#     with open('points.json', 'r') as file:
#         data = json.load(file)
#         puas_points = data.get('puas_points', 0)
#         tidak_puas_points = data.get('tidak_puas_points', 0)
# except FileNotFoundError:
#     pass

# video_running = False
        
# def save_expression_images(most_common_expression, face_images):
#     # Membuat nama folder untuk hasil ekspresi terbanyak
#     expression_dir = os.path.join(image_dir, most_common_expression)
    
#     # Mencari nama folder yang unik jika sudah ada
#     unique_expression_dir = expression_dir
#     index = 1
#     while os.path.exists(unique_expression_dir):
#         unique_expression_dir = f"{expression_dir}_{index}"
#         index += 1
    
#     # Membuat folder untuk menyimpan hasil ekspresi
#     os.makedirs(unique_expression_dir)
    
#     # Membuat nama unik untuk setiap gambar
#     timestamp = int(time.time() * 1000)
    
#     for i, face_image in enumerate(face_images):
#         image_path = os.path.join(unique_expression_dir, f"{timestamp}_{i}.jpg")
#         cv2.imwrite(image_path, face_image)

# # Fungsi untuk mengambil video dari kamera
# def gen_frames():
#         global video_running
#         cap = cv2.VideoCapture(1)  # Nomor 0 mengacu pada kamera default (biasanya kamera built-in)

#         start_time = time.time()
#         current_time = time.time()
#         image_count = 0 # Menyimpan hasil ekspresi pada setiap iterasi
#         face_images = [] # Menyimpan gambar wajah yang terdeteksi

#         while (current_time - start_time) < duration:
#             success, frame = cap.read()  # Membaca frame dari kamera
#             if not success:
#                 break
#             else:
#                 current_time = time.time()
#                 image_count += 1
                                
#                 # Konversi frame ke citra grayscale
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
#                     # Konversi citra grayscale ke citra RGB (ini tidak perlu)
#                 # face_roi_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

#                     # Deteksi wajah
#                 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

#                 # Untuk setiap wajah yang terdeteksi
#                 for (x, y, w, h) in faces:
#                     # Potong kotak area wajah dari citra
#                     face_roi = gray[y:y+h, x:x+w]
#                     #ini tidak perlu
#                     # face_roi = face_roi_rgb[y:y+h, x:x+w]
#                     face_roi = cv2.resize(face_roi, (48, 48))  # Rescale ke ukuran yang sesuai dengan model klasifikasi

#                     # Normalisasi piksel
#                     face_roi = face_roi / 255.0

#                     # Prediksi ekspresi
#                     expression_probs = emotion_model.predict(np.expand_dims(face_roi, axis=0))
#                     predicted_expression_label = np.argmax(expression_probs)

#                     # Menentukan label ekspresi yang sesuai
#                     expression_labels = ["Puas", "TidakPuas"]
#                     predicted_expression = expression_labels[predicted_expression_label]
#                     expression_results.append(predicted_expression)  # Menyimpan hasil ekspresi

#                     # Menyimpan gambar wajah yang terdeteksi
#                     face_images.append(frame[y:y+h, x:x+w])
                    
#                     # Hitung akurasi
#                     accuracy = expression_probs[0][predicted_expression_label] * 100

#                     # Menampilkan hasil prediksi pada layar
#                     text = f'Ekspresi: {predicted_expression} ({accuracy:.2f}%)'
#                     cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 frame = buffer.tobytes()

#                 yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Mengirim frame sebagai respons HTTP
# # Setelah selesai mengambil gambar, simpan gambar-gambar ke dalam folder
#         if face_images:
#             most_common_expression = Counter(expression_results).most_common(1)[0][0]
#             save_expression_images(most_common_expression, face_images)
#     # Setelah selesai mengambil gambar, hitung ekspresi terbanyak
#             expression_count = Counter(expression_results)
#             most_common_expression = expression_count.most_common(1)
#             print (f'Ekspresi Terbanyak: {most_common_expression[0][0]}')
            
#             global puas_points, tidak_puas_points
#             if most_common_expression:
#                 most_common_expression = most_common_expression[0][0]
#                 if most_common_expression == "Puas":
#                     puas_points += 1
#                 elif most_common_expression == "TidakPuas":
#                     tidak_puas_points += 1
#                 # Simpan poin ke file setiap kali ada perubahan
#                 save_points_to_file()

#     # Mengirim hasil ekspresi terbanyak dan poin ke klien
#             socketio.emit('update_expression', {'most_common_expression': predicted_expression,
#                                     'puas_points': puas_points, 'tidak_puas_points': tidak_puas_points})

# def save_points_to_file():
#     data = {'puas_points': puas_points, 'tidak_puas_points': tidak_puas_points}
#     with open('points.json', 'w') as file:
#         json.dump(data, file)

# @app.route('/')
# def index():
#     return render_template('tidak_puas.html')
#     # return render_template('tentang.html', video_running=video_running, puas_points=puas_points, tidak_puas_points=tidak_puas_points)
    
# @app.route('/tentang')
# def tentang():
#     return render_template('tentang.html', video_running=video_running, puas_points=puas_points, tidak_puas_points=tidak_puas_points)

# # @app.route('/start_system', methods=['POST'])
# # def start_system():
# #     print('System started manually.')
# #     # Tambahkan logika untuk memulai sistem di sini
# #     return jsonify({'status': 'System started'})

# # # Rute untuk menghentikan sistem
# # @app.route('/stop_system', methods=['POST'])
# # def stop_system():
# #     print('System stopped manually.')
# #     # Tambahkan logika untuk menghentikan sistem di sini
# #     return jsonify({'status': 'System stopped'})

# @socketio.on('get_expression')
# def get_expression():
#     expression_count = Counter(expression_results)
#     most_common_expression = expression_count.most_common(1)

#     if most_common_expression:
#         most_common_expression = most_common_expression[0][0]
#     else:
#         most_common_expression = "Tidak ada hasil ekspresi yang terdeteksi"
#     print(f"Most Common Expression: {most_common_expression}")
#     socketio.emit('update_expression', {'most_common_expression': most_common_expression, 'puas_points': puas_points, 'tidak_puas_points': tidak_puas_points})

# @app.route('/get_points')
# def get_points():
#     return jsonify({'puas': puas_points, 'tidak_puas': tidak_puas_points})


# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#     # return Response(mimetype='multipart/x-mixed-replace; boundary=frame')

# # if __name__ == '__main__':
# #     app.run(debug=True)
    
# if __name__ == '__main__':
#     socketio.run(app, debug=True)

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import time
import numpy as np
from keras.models import model_from_json
# from keras.models import load_model
from collections import Counter
import json
import os
from keras import backend as K

# import eventlet

app = Flask(__name__)
socketio = SocketIO(app)

json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load model klasifikasi ekspresi wajah (gantilah dengan jalur yang sesuai)
emotion_model.load_weights("emotion_model.h5")

# # Load detektor wajah (misalnya, menggunakan OpenCV Haar Cascades)
face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

duration = 30
image_interval = 1

expression_results = []
puas_points = 0
tidak_puas_points = 0

# Direktori untuk menyimpan gambar
image_dir = "expression_images"

# Fungsi untuk membuat folder jika belum ada
def create_image_folder():
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

# Panggil fungsi untuk membuat folders
create_image_folder()

# Baca poin ekspresi dari file jika ada
try:
    with open('points.json', 'r') as file:
        data = json.load(file)
        puas_points = data.get('puas_points', 0)
        tidak_puas_points = data.get('tidak_puas_points', 0)
except FileNotFoundError:
    pass

video_running = False
        
def save_expression_images(most_common_expression, face_images):
    # Membuat nama folder untuk hasil ekspresi terbanyak
    expression_dir = os.path.join(image_dir, most_common_expression)
    
    # Mencari nama folder yang unik jika sudah ada
    unique_expression_dir = expression_dir
    index = 1
    while os.path.exists(unique_expression_dir):
        unique_expression_dir = f"{expression_dir}_{index}"
        index += 1
    
    # Membuat folder untuk menyimpan hasil ekspresi
    os.makedirs(unique_expression_dir, exist_ok=True)
    
    # Membuat nama unik untuk setiap gambar
    timestamp = int(time.time() * 1000)
    
    for i, face_image in enumerate(face_images):
        # Menentukan hasil prediksi ekspresi untuk gambar wajah ini
        predicted_expression = expression_results[i]
        # Membuat nama file berdasarkan hasil prediksi ekspresi
        image_path = os.path.join(unique_expression_dir, f"{predicted_expression}_{timestamp}_{i}.jpg")
        # Menyimpan gambar dengan nama file yang sesuai dengan prediksi ekspresinya
        cv2.imwrite(image_path, face_image)
        
        # Tambahkan pernyataan print untuk memeriksa nilai predicted_expression dan most_common_expression
        print(f"Predicted Expression: {predicted_expression}, Most Common Expression: {most_common_expression}")
    # for i, face_image in enumerate(face_images):
    #     image_path = os.path.join(unique_expression_dir, f"{timestamp}_{i}.jpg")
    #     cv2.imwrite(image_path, face_image)

# Fungsi untuk mengambil video dari kamera
def gen_frames():
        global video_running
        global expression_results  # Tambahkan baris ini
        cap = cv2.VideoCapture(1)  # Nomor 0 mengacu pada kamera default (biasanya kamera built-in)
        expression_results = []
        start_time = time.time()
        current_time = time.time()
        image_count = 0 # Menyimpan hasil ekspresi pada setiap iterasi
        face_images = [] # Menyimpan gambar wajah yang terdeteksi

        while (current_time - start_time) < duration:
            success, frame = cap.read()  # Membaca frame dari kamera
            if not success:
                break
            else:
                current_time = time.time()
                if (current_time - start_time) >= (image_count * image_interval):
                    image_count += 1

                    # Konversi frame ke citra grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Deteksi wajah
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

                    # Untuk setiap wajah yang terdeteksi
                    for (x, y, w, h) in faces:
                        # Potong kotak area wajah dari citra
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (48, 48))  # Rescale ke ukuran yang sesuai dengan model klasifikasi

                        # Menyimpan gambar wajah yang terdeteksi
                        face_images.append(face_roi)

                        # Normalisasi piksel
                        face_roi = face_roi / 255.0

                        # Prediksi ekspresi
                        expression_probs = emotion_model.predict(np.expand_dims(face_roi, axis=0))
                        predicted_expression_label = np.argmax(expression_probs)

                        # Menentukan label ekspresi yang sesuai
                        expression_labels = ["Puas", "TidakPuas"]
                        predicted_expression = expression_labels[predicted_expression_label]
                        expression_results.append(predicted_expression)  # Menyimpan hasil ekspresi

                        # Hitung akurasi
                        accuracy = expression_probs[0][predicted_expression_label] * 100

                        # Menampilkan hasil prediksi pada layar
                        text = f'Ekspresi: {predicted_expression} ({accuracy:.2f}%)'
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print(f'Expression Results: {expression_results}')  # Tambahkan baris ini untuk memeriksa hasil prediksi

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Mengirim frame sebagai respons HTTP

        # Setelah selesai mengambil gambar, simpan gambar-gambar ke dalam folder
        if face_images:
            most_common_expression = Counter(expression_results).most_common(1)[0][0]
            save_expression_images(most_common_expression, face_images)
    # Setelah selesai mengambil gambar, hitung ekspresi terbanyak
            expression_count = Counter(expression_results)
            most_common_expression = expression_count.most_common(1)
            print (f'Ekspresi Terbanyak: {most_common_expression[0][0]}')

            global puas_points, tidak_puas_points
            if most_common_expression:
                most_common_expression = most_common_expression[0][0]
                if most_common_expression == "Puas":
                    puas_points += 1
                elif most_common_expression == "TidakPuas":
                    tidak_puas_points += 1
                # Simpan poin ke file setiap kali ada perubahan
                save_points_to_file()

            # Mengirim hasil ekspresi terbanyak dan poin ke klien
            socketio.emit('update_expression', {'most_common_expression': most_common_expression,
                                                'puas_points': puas_points, 'tidak_puas_points': tidak_puas_points})
#                 image_count += 1
                                
#                 # Konversi frame ke citra grayscale
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
#                     # Konversi citra grayscale ke citra RGB (ini tidak perlu)
#                 # face_roi_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

#                     # Deteksi wajah
#                 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

#                 # Untuk setiap wajah yang terdeteksi
#                 for (x, y, w, h) in faces:
#                     # Potong kotak area wajah dari citra
#                     face_roi = gray[y:y+h, x:x+w]
#                     #ini tidak perlu
#                     # face_roi = face_roi_rgb[y:y+h, x:x+w]
#                     face_roi = cv2.resize(face_roi, (48, 48))  # Rescale ke ukuran yang sesuai dengan model klasifikasi
 
#                     # Menyimpan gambar wajah yang terdeteksi
#                     face_images.append(face_roi)
                    
#                     # Normalisasi piksel
#                     face_roi = face_roi / 255.0

#                     # Prediksi ekspresi
#                     expression_probs = emotion_model.predict(np.expand_dims(face_roi, axis=0))
#                     predicted_expression_label = np.argmax(expression_probs)

#                     # Menentukan label ekspresi yang sesuai
#                     expression_labels = ["Puas", "TidakPuas"]
#                     predicted_expression = expression_labels[predicted_expression_label]
#                     expression_results.append(predicted_expression)  # Menyimpan hasil ekspresi


                    
#                     # Hitung akurasi
#                     accuracy = expression_probs[0][predicted_expression_label] * 100

#                     # Menampilkan hasil prediksi pada layar
#                     text = f'Ekspresi: {predicted_expression} ({accuracy:.2f}%)'
#                     cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 frame = buffer.tobytes()

#                 yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Mengirim frame sebagai respons HTTP
# # Setelah selesai mengambil gambar, simpan gambar-gambar ke dalam folder
#         if face_images:
#             most_common_expression = Counter(expression_results).most_common(1)[0][0]
#             save_expression_images(most_common_expression, face_images)
#     # Setelah selesai mengambil gambar, hitung ekspresi terbanyak
#             expression_count = Counter(expression_results)
#             most_common_expression = expression_count.most_common(1)
#             print (f'Ekspresi Terbanyak: {most_common_expression[0][0]}')
            
#             global puas_points, tidak_puas_points
#             if most_common_expression:
#                 most_common_expression = most_common_expression[0][0]
#                 if most_common_expression == "Puas":
#                     puas_points += 1
#                 elif most_common_expression == "TidakPuas":
#                     tidak_puas_points += 1
#                 # Simpan poin ke file setiap kali ada perubahan
#                 save_points_to_file()

#     # Mengirim hasil ekspresi terbanyak dan poin ke klien
#             socketio.emit('update_expression', {'most_common_expression': predicted_expression,
#                                     'puas_points': puas_points, 'tidak_puas_points': tidak_puas_points})

def save_points_to_file():
    data = {'puas_points': puas_points, 'tidak_puas_points': tidak_puas_points}
    with open('points.json', 'w') as file:
        json.dump(data, file)

@app.route('/')
def index():
    return render_template('tidak_puas.html')
    # return render_template('tentang.html', video_running=video_running, puas_points=puas_points, tidak_puas_points=tidak_puas_points)
    
@app.route('/tentang')
def tentang():
    return render_template('tentang.html', video_running=video_running, puas_points=puas_points, tidak_puas_points=tidak_puas_points)

# @app.route('/start_system', methods=['POST'])
# def start_system():
#     print('System started manually.')
#     # Tambahkan logika untuk memulai sistem di sini
#     return jsonify({'status': 'System started'})

# # Rute untuk menghentikan sistem
# @app.route('/stop_system', methods=['POST'])
# def stop_system():
#     print('System stopped manually.')
#     # Tambahkan logika untuk menghentikan sistem di sini
#     return jsonify({'status': 'System stopped'})

@socketio.on('get_expression')
def get_expression():
    expression_count = Counter(expression_results)
    most_common_expression = expression_count.most_common(1)

    if most_common_expression:
        most_common_expression = most_common_expression[0][0]
    else:
        most_common_expression = "Tidak ada hasil ekspresi yang terdeteksi"
    print(f"Most Common Expression: {most_common_expression}")
    socketio.emit('update_expression', {'most_common_expression': most_common_expression, 'puas_points': puas_points, 'tidak_puas_points': tidak_puas_points})

@app.route('/get_points')
def get_points():
    return jsonify({'puas': puas_points, 'tidak_puas': tidak_puas_points})


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)
    
if __name__ == '__main__':
    socketio.run(app, debug=True)