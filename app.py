import streamlit as st
import requests
import os
from streamlit_mic_recorder import mic_recorder

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


def query(data):
    # with open(filename, "rb") as f:
    #     data = f.read()
    response = requests.post(API_URL, headers={"Content-Type": "audio/wave"}, data=data)
    return response.json()

# def start_recording():
#     global recording
#     if not recording:   
#         recording = True
#         st.write("Recording started...")
#         fs = 48000  
#         myrecording = sd.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True)  
#         global audio_frames
#         audio_frames = []
        
#         while recording:
#             audio_frames.append(myrecording.read(1024))

# def stop_recording():
#     if recording:    
#         recording = False
#         st.write("Recording stopped")
#         sd.wait()  

#         # st.audio(myrecording.tobytes(), sample_rate=fs) 

#         # # wav_file = open('output.wav','rb')
#         # # bytes_data = wav_file.read()

#         # recording = False
#         # st.write("Recording stopped")
        
#         # myrecording.close()
        
#         full_recording = b''.join([f for f in audio_frames]) 
        
#         output = query(full_recording.tobytes())
#         value = output['text']  
#         st.markdown(f'<p class="big-font">{value}</p>', unsafe_allow_html=True)

def file_inference():
        global show_record_buttons
        show_record_buttons = False
        uploaded_file = st.file_uploader("Upload audio file") 
        if uploaded_file:
            output = query(uploaded_file.read())
            value = output['text']
            st.markdown(f'<p class="big-font">{value}</p>', unsafe_allow_html=True)

tabs = ["Sign Language", "Arabic Speech-to-Text Uploaded", "Arabic Speech-to-Text Recording"]
tab = st.sidebar.radio("Select tab", tabs)
os.environ['CURL_CA_BUNDLE'] = ''

if tab == "Sign Language":
    
    st.header("Sign Language Recognition")
    
    st.write("Coming soon...")
    
elif tab == "Arabic Speech-to-Text Uploaded":
    if os.path.exists('myfile.wav'):
        os.remove('myfile.wav')
    st.title('Upload wav/mp3 file, and get Arabic Speech-to-text!')
    file_inference()

elif tab == "Arabic Speech-to-Text Recording":
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
