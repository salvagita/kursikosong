import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time

st.set_page_config(layout="wide")

cfg_model_path = 'yolov5s.pt'
model = None
confidence = 0.25
selected_classes = None


CLASSES = [
    'Kursi_Terisi', 'Kursi_Kosong'
]

def video_input(data_src, input_option):
    vid_file = None
    if data_src == 'Video':
        vid_bytes = st.sidebar.file_uploader("Unggah video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())
            st.sidebar.video(vid_bytes)
            if not os.path.exists('videos'):
                os.makedirs('videos')
            with open(os.path.join('videos', vid_bytes.name), 'wb') as f:
                f.write(vid_bytes.getbuffer())
    elif data_src == 'Real-time':
        vid_file = 'Real-time'

    if vid_file:
        if vid_file == 'Real-time':
            cap = cv2.VideoCapture(1)
        else:
            cap = cv2.VideoCapture(vid_file)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st1, st2 = st.columns(2)
        with st1:
            st.markdown("## Kursi Kosong")
            st1_Kursi_Kosong_count = st.markdown("__")
        with st2:
            st.markdown("## Kursi Terisi")
            st2_Kursi_Terisi_count = st.markdown("__")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        video_started = False

        if st.sidebar.button("Mulai"):
            video_started = True

        while video_started:
            ret, frame = cap.read()
            
            if not ret:
                st.write("Tidak dapat membaca frame, akhir stream? Keluar ....")
                break
            if data_src == 'Real-time':
                frame = cv2.flip(frame, 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img, Kursi_Terisi_count, Kursi_Kosong_count = infer_image(frame, None, confidence)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_Kursi_Kosong_count.markdown(f"**{Kursi_Kosong_count}**")
            st2_Kursi_Terisi_count.markdown(f"**{Kursi_Terisi_count}**")

        cap.release()



def infer_image(img, size=None, confidence=0.25):
    model.conf = confidence

    # Hitung gain (g) menggunakan faktor skala g = min(width, height) / 640
    g = min(img.shape[1], img.shape[0]) / 224.0
    
    if size:
        img = cv2.resize(img, None, fx=g, fy=g)

    result = model(img)
    result = result.pandas().xyxy[0]
    result = result[result['name'].isin(selected_classes)]
    result = result[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']]
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    Kursi_Terisi_count = sum(1 for _, detection in result.iterrows() if detection['name'] == 'Kursi_Terisi')
    Kursi_Kosong_count = sum(1 for _, detection in result.iterrows() if detection['name'] == 'Kursi_Kosong')
    
    for _, row in result.iterrows():
        label = row['name']
        confidence = row['confidence']
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        
        if label == 'Kursi_Terisi':
            color = (0, 0, 255)
        if label == 'Kursi_Kosong':
            color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # Ubah font dan ukuran teks
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.9
        
        # Ubah warna teks menjadi putih
        text_color = (255, 255, 255)
        
        # Gambar teks pada gambar
        cv2.putText(image, f"{confidence:.2f}", (xmin + 5, ymin - 5), font, font_size, text_color, 2, cv2.LINE_AA)
        
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image, Kursi_Terisi_count, Kursi_Kosong_count


@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model ke ", device)
    return model_


def main():
    global selected_classes, confidence, person_count

    # variabel global
    global model, cfg_model_path

    st.title("Deteksi Kursi Kosong Dan Terisi")

    st.sidebar.title("Pengaturan")

    # periksa apakah file model tersedia
    if not os.path.isfile(cfg_model_path):
        st.warning("File model tidak tersedia!!!, mohon tambahkan ke folder model.", icon="⚠️")
    else:
        
        # muat model
        model = load_model(cfg_model_path, 'cpu')

        # slider kepercayaan
        confidence = st.sidebar.slider('Kepercayaan', min_value=0.1, max_value=1.0, value=0.50)

        st.sidebar.markdown("---")

        # Pilihan objek yang ingin dideteksi
        st.sidebar.markdown("## Objek yang ingin dideteksi")
        selected_classes = st.sidebar.multiselect("Pilih Objek", CLASSES, default=["Kursi_Kosong"])

        st.sidebar.markdown("---")

        # Pilihan data sumber
        input_option = st.sidebar.selectbox("Sumber Data", ["Video", "Real-time"])

        if input_option:
            video_input(input_option, input_option)  # memperbarui argumen input_option

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
