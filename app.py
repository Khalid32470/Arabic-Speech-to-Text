import streamlit as st
import requests
import os
from streamlit_mic_recorder import mic_recorder
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from google.cloud import storage

st.set_page_config(page_title="My App", page_icon="🤖")
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

API_URL = "https://api-inference.huggingface.co/models/Salama1429/KalemaTech-Arabic-STT-ASR-based-on-Whisper-Small" 

headers = {"Authorization": "Bearer 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"}

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Actions that we try to detect
actions = np.array(['Assalamu Alaikum','Bayt','Yufham','Shukran laka','Khata','Maa','Jamiah','Madrasah','Aayen','Mushkilah','`'])

colors = [
    (255, 0, 0),     # Color for 'Assalamu Alaikum' - Blue
    (0, 255, 0),     # Color for 'Bayt' - Green
    (0, 0, 255),     # Color for 'Yufham' - Red
    (255, 255, 0),   # Color for 'Shukran laka' - Cyan
    (255, 0, 255),   # Color for 'Khata' - Magenta
    (0, 255, 255),   # Color for 'Maa' - Yellow
    (192, 192, 192), # Color for 'Jamiah' - Silver
    (128, 0, 0),     # Color for 'Madrasah' - Maroon
    (128, 128, 0),   # Color for 'Aayen' - Olive
    (0, 128, 0),     # Color for 'Mushkilah' - Dark Green
    (0, 0, 0),       # Color for 'No Action Detected' - Black
]

def load_model_from_gcs(bucket_name, model_name, local_path):

    if not os.path.exists(local_path):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(model_name)
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Model downloaded from GCS to {local_path}")

def query(data):
    # with open(filename, "rb") as f:
    #     data = f.read()
    response = requests.post(API_URL, headers={"Content-Type": "audio/wave"}, data=data)
    return response.json()

def file_inference():
        global show_record_buttons
        show_record_buttons = False
        uploaded_file = st.file_uploader("Upload audio file") 
        if uploaded_file:
            output = query(uploaded_file.read())
            value = output['text']
            st.markdown(f'<p class="big-font">{value}</p>', unsafe_allow_html=True)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    # Adjust these values if necessary
    bar_width = 100  # Maximum width of the bar for probability
    bar_height = 35  # Height of the bar
    padding = 10     # Space between bars
    
    # Start drawing from this y position
    start_y_position = 40  # You might need to adjust this if the first label is not showing

    # Ensure you have a color for each action
    colors = [colors[i % len(colors)] for i in range(len(actions))]

    # Iterate over all actions and probabilities
    for num, (prob, action) in enumerate(zip(res, actions)):
        bar_length = int(prob * bar_width)  # Scale the bar length by the probability

        # Determine the y position for the current bar
        bar_y_position = start_y_position + num * (bar_height + padding)

        # Draw the rectangle for the bar
        cv2.rectangle(output_frame, (0, bar_y_position), (bar_length, bar_y_position + bar_height), colors[num], -1)
        
        # Put the action text
        cv2.putText(output_frame, action, (10, bar_y_position + bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame

tabs = ["Sign Language", "Arabic Speech-to-Text Uploaded", "Arabic Speech-to-Text Recording"]
tab = st.sidebar.radio("Select tab", tabs, key='tab')
os.environ['CURL_CA_BUNDLE'] = ''

load_model_from_gcs('asl-model-weights', 'action_arabic_roman.h5', '/usr/app/models/action_arabic_roman.h5')

with st.container():
    placeholder = st.empty()
    if tab == "Sign Language":
        placeholder.empty()
        st.header("Sign Language Recognition")
        
        # Load the gesture recognition model
        model = load_model('/usr/app/models/action_arabic_roman.h5')
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)

        # New detection variables
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.7

        # Init Streamlit video widget
        vid = st.video(None)
        
        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Set the window size
            # Init OpenCV window
            cv2.namedWindow('SLR', cv2.WINDOW_NORMAL) 
            
            while cap.isOpened():
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # keep last 30 frames
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
                    
                    # Viz logic
                    if np.mean(predictions[-15:]) == np.argmax(res):  # Check if the last 10 predictions are stable
                        if res[np.argmax(res)] > threshold:  # Check if the highest prediction is above the threshold
                            if len(sentence) > 0 and actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                            elif len(sentence) == 0:
                                sentence.append(actions[np.argmax(res)])
                                
                    if len(sentence) > 3:  # Limit the sentence to last 5 words
                        sentence = sentence[-3:]

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
                
                cv2.rectangle(image, (0, 0), (1200, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Convert color for Streamlit
                display_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Show to screen
                vid.image(display_frame)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        
    elif tab == "Arabic Speech-to-Text Uploaded":
        placeholder.empty()
        if os.path.exists('myfile.wav'):
            os.remove('myfile.wav')
        st.title('Upload wav/mp3 file, and get Arabic Speech-to-text!')
        file_inference()

    elif tab == "Arabic Speech-to-Text Recording":
            placeholder.empty()
            if os.path.exists('myfile.wav'):
                os.remove('myfile.wav')
            # if st.button("Start recording"):
            st.title('Record your voice, and get Arabic Speech-to-text!')
            try:
                audio = mic_recorder(start_prompt="Start Recording",stop_prompt="Stop Recording",key='recorder')
                if audio:
                    st.audio(audio['bytes']) 
                    with open('myfile.wav', mode='bx') as f:
                        f.write(audio['bytes'])      
                    output = query(open('myfile.wav', mode='rb').read())
                    print(output)
                    value = output['text']
                    if os.path.exists('myfile.wav'):
                        os.remove('myfile.wav')
                    st.markdown(f'<p class="big-font">{value}</p>', unsafe_allow_html=True)
            except:
                if os.path.exists('myfile.wav'):
                    os.remove('myfile.wav')
                st.markdown(f'<p class="big-font">Something went wrong. Please try again.</p>', unsafe_allow_html=True)
