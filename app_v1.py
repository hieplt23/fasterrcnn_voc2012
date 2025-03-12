import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from argparse import ArgumentParser
import os
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, \
    FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
import base64
from io import BytesIO
import time


def get_args():
    parser = ArgumentParser(description='FasterR-CNN for pascal VOC data')
    parser.add_argument("--image_path", type=str, default='./data/test_images/test_1.jpg', help='root of dataset')
    parser.add_argument("--image_size", type=int, default=416, help='size of image')
    parser.add_argument("--num_classes", type=int, default=21, help='number of classes')
    parser.add_argument("--checkpoint_path", "-c", type=str, default='./trained_models/best.pt',
                        help='path of trained model')
    args = parser.parse_args()
    return args


def load_model(checkpoint_path, device, num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features,
                                                      num_classes=num_classes)
    model.to(device)
    if not os.path.isfile(checkpoint_path):
        print('Not found checkpoint!')
        exit(0)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model

def detect_objects(original_image, model, device, image_size, conf_threshold=0.2):
    categories = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    ]

    old_h, old_w, _ = original_image.shape
    test_image = cv2.resize(original_image, (image_size, image_size))
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB).astype(np.float64)
    test_image /= 255.
    test_image -= np.array([0.485, 0.456, 0.406])
    test_image /= np.array([0.229, 0.224, 0.225])
    test_image = test_image.transpose(2, 0, 1)
    test_image = [torch.FloatTensor(test_image).to(device)]

    model.eval()
    with torch.no_grad():
        results = model(test_image)

    results = results[0]
    boxes = results['boxes']
    labels = results['labels']
    scores = results['scores']

    indices = nms(boxes, scores, 0.5)
    for indice in indices:
        if scores[indice] > conf_threshold:
            xtl, ytl, xbr, ybr = boxes[indice]
            xtl = int(float(xtl.item()) / image_size * old_w)
            ytl = int(float(ytl.item()) / image_size * old_h)
            xbr = int(float(xbr.item()) / image_size * old_w)
            ybr = int(float(ybr.item()) / image_size * old_h)
            label_confident_text = f'{categories[labels[indice]]} {scores[indice]:.2f}'

            cv2.rectangle(original_image, (xtl, ytl), (xbr, ybr), color=(0, 255, 0), thickness=2)

            font_scale = 1
            font_thickness = 2
            font = cv2.FONT_ITALIC
            text_size, baseline = cv2.getTextSize(label_confident_text, font, font_scale, font_thickness)
            text_w, text_h = text_size

            text_x = xtl
            text_y = ytl - 10 if ytl > 10 else 10
            rect_xtl = text_x - 5
            rect_ytl = text_y - text_h - 5
            rect_xbr = text_x + text_w + 5
            rect_ybr = text_y + 5

            bg_color = (220, 220, 220)
            alpha = 0.6

            overlay = original_image.copy()
            cv2.rectangle(overlay, (rect_xtl, rect_ytl), (rect_xbr, rect_ybr), bg_color, -1)

            cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0, original_image)

            cv2.putText(original_image, label_confident_text, (text_x, text_y),
                        fontFace=font, fontScale=font_scale, color=(127, 0, 255), thickness=font_thickness)

    return original_image


def get_image_download_link(img_array, filename="detected_image.jpg"):
    pil_img = Image.fromarray(img_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">Download Result</a>'
    return href


# CSS
def load_css():
    st.markdown("""
    <style>
    .stApp { background-color: #f5f7fa; font-family: 'Segoe UI', sans-serif; }
    .main-title { font-size: 2.5em; color: #2c3e50; text-align: center; padding: 20px 0; font-weight: 600; }
    h3 { color: #34495e; font-weight: 500; margin-top: 20px; }
    .stButton>button { background-color: #3498db; color: white; border-radius: 8px; padding: 10px 20px; font-size: 16px; transition: all 0.3s; }
    .stButton>button:hover { background-color: #2980b9; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .stFileUploader { border: 2px dashed #bdc3c7; border-radius: 10px; padding: 20px; background-color: white; }
    .image-container { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .footer { text-align: center; color: #7f8c8d; padding: 20px 0; font-size: 0.9em; }
    .stSuccess { background-color: #e8f5e9; color: #2e7d32; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)


def app():
    args = get_args()
    load_css()

    # Header
    st.markdown('<div class="main-title">FasterRCNN Object Detection 🔍</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Powered by Pascal VOC 2012 Dataset</p>',
                unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    input_mode = st.sidebar.selectbox("Input Mode", ["Image Upload", "Camera/Video"])
    # image_size = st.sidebar.slider("Image Size", 224, 1024, 320, step=32)  # Giảm về 320 để tối ưu tốc độ
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.2, step=0.05)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint_path, device, args.num_classes)

    if input_mode == "Image Upload":
        st.write("### 📤 Upload Images")
        uploaded_files = st.file_uploader(
            "Supported formats: JPG, PNG, JPEG (Multiple files allowed)",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key="uploader"
        )

        if uploaded_files:
            st.write(f"**{len(uploaded_files)} image(s) uploaded**")
            num_cols = min(2, len(uploaded_files))
            cols = st.columns(num_cols)

            if st.button("✨ Detect Objects on All Images", key="detect_btn"):
                results = []
                with st.spinner("Processing images... Please wait"):
                    for uploaded_file in uploaded_files:
                        image = Image.open(uploaded_file)
                        img_array = np.array(image)
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        result_img = detect_objects(img_array, model, device, args.image_size, conf_threshold)
                        results.append((uploaded_file.name, image, result_img))

                for idx, (filename, orig_img, result_img) in enumerate(results):
                    col_idx = idx % num_cols
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    with cols[col_idx]:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(orig_img, caption=f"Original: {filename}", use_container_width=True)
                        st.image(result_img, caption=f"Result: {filename}", use_container_width=True)
                        st.markdown(get_image_download_link(result_img, filename), unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                st.success(f"Detection completed for {len(results)} image(s)! 🎉")

    elif input_mode == "Camera/Video":
        st.write("### 📹 Real-Time Object Detection")
        run_camera = st.checkbox("Start Camera", key="run_camera")

        video_frame = st.empty()

        if run_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access camera. Please check your webcam.")
                return

            st.info("Camera is running. Uncheck 'Start Camera' to stop.")
            frame_count = 0
            start_time = time.time()

            while run_camera and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera.")
                    break

                # Xử lý frame với object detection
                result_frame = detect_objects(frame, model, device, args.image_size, conf_threshold)

                # Hiển thị FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:  # Cập nhật FPS mỗi giây
                    fps = frame_count / elapsed_time
                    cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame_count = 0
                    start_time = time.time()

                # Chuyển frame sang RGB để hiển thị trong Streamlit
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                video_frame.image(result_frame_rgb, caption="Real-Time Detection", use_container_width=True)

                # Kiểm tra trạng thái checkbox
                run_camera = st.session_state.run_camera

            cap.release()

    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="footer">Created by @hieplt23 | Last updated: March 11, 2025</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    app()