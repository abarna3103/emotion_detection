import streamlit as st
import cv2
import numpy as np
from PIL import Image
import dlib
import torch
from torchvision import transforms
import torch.nn as nn
from io import BytesIO
import base64

# --- Enhanced UI Styling ---
st.set_page_config(page_title="Emotion Detection App", page_icon=":smiley:")

st.markdown(
    """
    <style>
        /* Background color */
        .stApp {
            background-color: #121212;
        }
        
        /* Text color */
        body, p, label, li {
        color: #e0e0e0 !important;  /* Light gray for better contrast */
        }

        /* Heading colors */
        h1, h2, h3, h4, h5, h6 {
            color: #ffcc00 !important;
        }

        /* Style buttons */
        .stButton>button {
            background-color: #ff6600 !important;
            color: white !important;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #cc5500 !important;
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)



# --- Model and Face Detection Logic ---
class EmotionCNN(nn.Module): # ... (Model definition remains the same)
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    model = EmotionCNN()
    model.load_state_dict(torch.load(r"C:\Users\Public\GUVI\code\face emotion detection\emotion-detection-streamlit\emotion_cnn_model.pth", map_location=torch.device('cpu')))  # ***REPLACE WITH YOUR MODEL PATH***
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

detector = dlib.get_frontal_face_detector()

def detect_face(image): # ... (Face detection remains the same)
    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if len(faces) == 0:
        return image, None
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image, faces

def classify_emotion(face_image_rgb): # ... (Emotion classification remains the same)
    face_image_gray = face_image_rgb.convert("L")
    face_image_gray = transform(face_image_gray).unsqueeze(0)
    with torch.no_grad():
        output = model(face_image_gray)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def image_to_bytes(image): # ... (Image to bytes for download remains the same)
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def process_frame(frame): # ... (Process Frame remains the same)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_np = np.array(image)

    detected_image, faces = detect_face(image_np.copy())

    if faces:
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_image_rgb = Image.fromarray(image_np[y:y+h, x:x+w])
            predicted_emotion = classify_emotion(face_image_rgb)

            emotion_labels = { # ... (emotion labels remain the same)
                0: ("Anger", "😡"),
                1: ("Disgust", "🤢"),
                2: ("Fear", "😱"),
                3: ("Happiness", "😊"),
                4: ("Sadness", "😢"),
                5: ("Surprise", "😲"),
                6: ("Neutral", "😐")
            }
            predicted_emotion_label, emoji = emotion_labels[predicted_emotion]

            cv2.putText(detected_image, f"{predicted_emotion_label} {emoji}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            emotion_details = { # ... (emotion details remain the same)
                "Anger": "A strong feeling of displeasure or hostility.",
                "Happiness": "A feeling of joy, contentment, or well-being.",
                "Sadness": "A feeling of sorrow or unhappiness.",
                "Fear": "A feeling of dread or anxiety.",
                "Disgust": "A strong feeling of dislike or disapproval.",
                "Surprise": "A feeling of astonishment or shock.",
                "Neutral": "A state of being indifferent or without strong emotion."
            }
            # You can add the emotion_details display here if you want it on the webcam feed as well.

    return detected_image


# --- Streamlit App ---
st.markdown("<h1>Emotion Detection App</h1>", unsafe_allow_html=True)  # Main title

st.markdown("""
<h2>How It Works</h2>
<ol>
    <li><b>Choose an Option:</b> Select either "Upload Image" or "Webcam" to proceed.</li>
    <li><b>Upload an Image / Use Webcam:</b> If uploading, select a clear image (JPG, JPEG, PNG, max 5MB). If using the webcam, allow access.</li>
    <li><b>Detect Emotion:</b> Click the "Detect Emotion" button (for images) or see the real-time predictions (for webcam).</li>
    <li><b>Emotion Results:</b> The detected emotion will be displayed with an emoji.</li>
    <li><b>Download Image:</b> You can download the processed image.</li>
</ol>
<p>If no face is detected, ensure the image/webcam feed contains a visible face.</p>
""", unsafe_allow_html=True)  # Instructions with ordered list

app_mode = st.selectbox("Choose an option", ["Upload Image", "Webcam"])

if app_mode == "Upload Image":
    st.markdown("<h3>Upload Your Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    MAX_FILE_SIZE_MB = 5
    MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024

    if uploaded_file is not None:
        try:
            file_size = uploaded_file.size
            if file_size > MAX_FILE_SIZE:
                st.warning(f"File size exceeds {MAX_FILE_SIZE_MB} MB. Please upload a smaller image.")
            else:
                image = Image.open(uploaded_file)
                image_np = np.array(image)

                if image_np.shape[1] > 600:
                    image_np = cv2.resize(image_np, (600, int(image_np.shape[0] * (600 / image_np.shape[1]))))

                st.image(image, caption="Uploaded Image", use_container_width=True)  # Display uploaded image

                if st.button("Detect Emotion"):  # Process and display when button clicked
                    with st.spinner('Processing...'):  # Show spinner while processing
                        processed_image, faces = detect_face(image_np.copy())  # Detect faces

                        if faces:
                            for face in faces:
                                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                                face_image_rgb = Image.fromarray(image_np[y:y+h, x:x+w])
                                predicted_emotion = classify_emotion(face_image_rgb)

                                emotion_labels = { # ... (emotion labels remain the same)
                                    0: ("Anger", "😡"),
                                    1: ("Disgust", "🤢"),
                                    2: ("Fear", "😱"),
                                    3: ("Happiness", "😊"),
                                    4: ("Sadness", "😢"),
                                    5: ("Surprise", "😲"),
                                    6: ("Neutral", "😐")
                                }
                                predicted_emotion_label, emoji = emotion_labels[predicted_emotion]

                                st.markdown(f"<h2>{predicted_emotion_label} {emoji}</h2>", unsafe_allow_html=True)  # Display emotion

                                emotion_details = { # ... (emotion details remain the same)
                                    "Anger": "A strong feeling of displeasure or hostility.",
                                    "Happiness": "A feeling of joy, contentment, or well-being.",
                                    "Sadness": "A feeling of sorrow or unhappiness.",
                                    "Fear": "A feeling of dread or anxiety.",
                                    "Disgust": "A strong feeling of dislike or disapproval.",
                                    "Surprise": "A feeling of astonishment or shock.",
                                    "Neutral": "A state of being indifferent or without strong emotion."
                                }
                                st.write(f"**Emotion Details:** {emotion_details[predicted_emotion_label]}")

                            st.image(processed_image, caption='Processed Image with Face Detection', use_container_width=True)  # Display processed image

                            download_image = image_to_bytes(Image.fromarray(processed_image)) # Corrected: Use processed_image
                            st.download_button(label="Download Processed Image", data=download_image, file_name="processed_image.png", mime="image/png")
                        else:
                            st.write("No face detected in the image. Please upload an image with a visible face.")

        except Exception as e:
            st.error(f"Error processing the image: {e}")
elif app_mode == "Webcam":
    st.markdown("<h3>Webcam Emotion Detection</h3>", unsafe_allow_html=True)
    WEBCAM_WIDTH = 640
    WEBCAM_HEIGHT = 480

    video_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

        capture_button = st.button("Capture Frame", key="capture_button")  # Button created ONCE

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame)

            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            video_placeholder.image(pil_image, use_container_width=True)

            if capture_button:  # Check if the button was clicked
                download_image = image_to_bytes(pil_image)
                st.download_button(label="Download Captured Image", data=download_image, file_name="captured_image.png", mime="image/png")
                capture_button = False  # Reset the button state (important!)

        cap.release()