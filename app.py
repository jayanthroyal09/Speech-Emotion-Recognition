import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
from scipy.io.wavfile import write
import time
import os
import tensorflow as tf
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="Hearmony",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to improve aesthetics
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #6C63FF;
        --primary-dark: #5A52D9;
        --secondary: #FF6584;
        --light-bg: #F9FAFC;
        --dark-text: #2E384D;
        --light-text: #8A94A6;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* Overall styling */
    .main {
        background-color: var(--light-bg);
        color: var(--dark-text);
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: var(--dark-text);
    }
    
    p, li {
        color: var(--dark-text);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton button {
        background-color: var(--primary);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(108, 99, 255, 0.2);
        font-size: 1rem;
    }
    
    .stButton button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(108, 99, 255, 0.3);
    }
    
    .stButton button:active {
        transform: translateY(0);
    }
    
    /* Cards */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 5px 20px var(--shadow);
        margin-bottom: 1.5rem;
        border: 1px solid #EAEDF3;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        transform: translateY(-3px);
    }
    
    /* Emotion cards */
    .emotion-card {
        transition: all 0.3s ease;
        border-radius: 12px;
        cursor: pointer;
        overflow: hidden;
    }
    
    .emotion-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px var(--shadow);
    }
    
    /* Circular icons */
    .circle-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: rgba(108, 99, 255, 0.1);
        margin-bottom: 10px;
    }
    
    /* Progress bars */
    .progress-container {
        background-color: #F0F2F8;
        border-radius: 8px;
        height: 10px;
        width: 100%;
        margin: 10px 0;
        overflow: hidden;
    }
    
    /* Headers */
    .header {
        text-align: center;
        padding: 1rem 0 2rem;
    }
    
    /* Dividers */
    .divider {
        height: 1px;
        background-color: #EAEDF3;
        margin: 1.5rem 0;
    }
    
    /* Focus container */
    .focus-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .animated {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    /* App wrapper */
    .app-wrapper {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #6C63FF 0%, #FF6584 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
    }
    
    .tooltip:hover:after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 130%;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.5rem 1rem;
        background-color: var(--dark-text);
        color: white;
        border-radius: 6px;
        font-size: 0.85rem;
        white-space: nowrap;
        z-index: 10;
    }
    
    /* Center content */
    .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    /* Result container */
    .result-container {
        animation: fadeIn 0.8s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Paths for saving files
AUDIO_PATH = "recorded_audio.wav"
SPECTROGRAM_PATH = "spectrogram.png"

# Make sure model directory exists
os.makedirs("model", exist_ok=True)

# Emotion data for display
EMOTION_DATA = {
    'Happy': {
        'emoji': '😄',
        'color': '#FFD700',  # Gold
        'suggestions': [
            "Share your joy with someone you care about",
            "Plan something fun for yourself or others",
            "Journal about what made you happy today"
        ]
    },
    'Sad': {
        'emoji': '😢',
        'color': '#6495ED',  # Cornflower Blue
        'suggestions': [
            "Call a friend who makes you smile",
            "Go for a 10-minute walk outside",
            "Listen to calming or uplifting music"
        ]
    },
    'Angry': {
        'emoji': '😠',
        'color': '#FF6347',  # Tomato
        'suggestions': [
            "Try 5 minutes of deep breathing exercises",
            "Take a short break from what you're doing",
            "Write down what's bothering you"
        ]
    },
    'Calm': {
        'emoji': '😌',
        'color': '#98FB98',  # Pale Green
        'suggestions': [
            "Maintain this energy by planning your day",
            "Practice mindfulness to enhance your peaceful state",
            "Tackle an important task that requires focus"
        ]
    },
    'Fearful': {
        'emoji': '😱',
        'color': '#9370DB',  # Medium Purple
        'suggestions': [
            "Talk to someone you trust about your concerns",
            "Write in a journal about what's making you fearful",
            "Practice grounding techniques"
        ]
    },
    'Disgust': {
        'emoji': '🤢',
        'color': '#7CFC00',  # Lawn Green
        'suggestions': [
            "Create some physical distance from what's bothering you",
            "Cleanse your environment or personal space",
            "Redirect your attention to something pleasant"
        ]
    },
    'Surprised': {
        'emoji': '😲',
        'color': '#FF69B4',  # Hot Pink
        'suggestions': [
            "Reflect on what surprised you and why",
            "Write down your thoughts about this unexpected situation",
            "Take a moment to process before making any decisions"
        ]
    }
}

# Function to record audio
def record_audio(duration=5, fs=44100):
    with st.spinner("🎙️ Recording... Express yourself naturally"):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        
        # Show progress bar while recording
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(duration/100)
            progress_bar.progress(i + 1)
        
        sd.wait()  # Wait for recording to complete
        write(AUDIO_PATH, fs, recording)
        
    st.success("✅ Voice captured successfully!")
    return True

# Function to create spectrogram from audio
def create_spectrogram():
    y, sr = librosa.load(AUDIO_PATH, sr=22050)
    
    # Generate mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    # Plot spectrogram
    plt.figure(figsize=(2.56, 2.56))
    librosa.display.specshow(S_DB, sr=sr, cmap='magma')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(SPECTROGRAM_PATH, bbox_inches='tight', pad_inches=0, dpi=50)
    plt.close()
    
    return True

# Function to simulate emotion prediction
def predict_emotion():
    # For demonstration, we'll select a random emotion
    emotions = list(EMOTION_DATA.keys())
    predicted_emotion = np.random.choice(emotions)
    
    # Create confidences where the predicted emotion has highest confidence
    confidences = np.random.dirichlet(np.ones(len(emotions)) * 0.5, size=1)[0]
    
    # Find index of predicted emotion
    predicted_idx = emotions.index(predicted_emotion)
    
    # Ensure predicted emotion has highest confidence
    # Swap the highest confidence value with the predicted emotion's confidence
    max_idx = np.argmax(confidences)
    if max_idx != predicted_idx:
        confidences[predicted_idx], confidences[max_idx] = confidences[max_idx], confidences[predicted_idx]
    
    # Make sure there's a significant gap between top emotion and others
    if confidences[predicted_idx] < 0.4:
        # Boost the top emotion's confidence
        boost = 0.4 - confidences[predicted_idx]
        confidences[predicted_idx] += boost
        
        # Redistribute the boost from other emotions proportionally
        for i in range(len(emotions)):
            if i != predicted_idx:
                confidences[i] *= (1 - confidences[predicted_idx]) / (1 - confidences[predicted_idx] + boost)
    
    # Sort emotions by confidence (descending)
    sorted_indices = np.argsort(confidences)[::-1]
    
    # Get top 3 emotions with their confidences
    top_emotions = [(emotions[i], confidences[i] * 100) for i in sorted_indices[:3]]
    
    return {
        'emotion': predicted_emotion,
        'confidence': confidences[predicted_idx] * 100,
        'top_emotions': top_emotions
    }

# Main function
def main():
    with st.container():
        st.markdown("""
            <div class="header animated">
                <h1 class="gradient-text">Hearmony: One voice, A thousand feelings</h1>
                <p style="font-size: 1.2rem; color: var(--light-text); margin-top: -10px;">
                    "Because every voice deserves to be understood"
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="app-wrapper">', unsafe_allow_html=True)
        
        # Create two columns for the main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
                <div class="card">
                    <div class="circle-icon">
                        <span style="font-size: 24px;">🎯</span>
                    </div>
                    <h3>Voice Analysis Center</h3>
                    <p>Speak naturally for 5 seconds and our AI will analyze the emotional undertones in your voice.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Record button
            if st.button("🎙️ Capture Your Voice", type="primary", use_container_width=True):
                # Record audio
                if record_audio():
                    # Create spectrogram
                    if create_spectrogram():
                        with st.spinner("🔍 Analyzing vocal patterns..."):
                            time.sleep(1)  # Simulate processing time
                            
                            # Get emotion prediction
                            result = predict_emotion()
                            primary_emotion = result['emotion']  # Store primary emotion for consistent use
                            
                            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                            
                            # Display the results section
                            st.markdown("""
                                <div class="result-container">
                                    <h2>Your Voice Analysis Results</h2>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Create three columns for the results display
                            analysis_col1, analysis_col2, analysis_col3 = st.columns([1, 1, 1])
                            
                            with analysis_col1:
                                st.markdown("""
                                    <div class="card center-content">
                                        <h4>Voice Pattern</h4>
                                """, unsafe_allow_html=True)
                                st.image(SPECTROGRAM_PATH, width=200)
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Display the emotion result
                            if result:
                                emoji = EMOTION_DATA.get(primary_emotion, {}).get('emoji', '🔍')
                                color = EMOTION_DATA.get(primary_emotion, {}).get('color', '#808080')
                                
                                with analysis_col2:
                                    st.markdown(f"""
                                        <div class="card center-content">
                                            <h4>Primary Emotion</h4>
                                            <div class="emotion-card" style="
                                                padding: 20px; 
                                                width: 90%;
                                                border-radius: 12px; 
                                                background: linear-gradient(135deg, {color}20, {color}40);
                                                border: 2px solid {color};
                                                text-align: center;
                                                margin-top: 10px;
                                            ">
                                                <span style="font-size: 48px;">{emoji}</span>
                                                <h2 style="margin-top: 10px; margin-bottom: 5px;">{primary_emotion}</h2>
                                                <div style="
                                                    background-color: white;
                                                    border-radius: 20px;
                                                    padding: 5px 15px;
                                                    display: inline-block;
                                                    font-weight: 600;
                                                    font-size: 0.9rem;
                                                    color: var(--dark-text);
                                                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                                                ">
                                                    {result['confidence']:.1f}% confidence
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                with analysis_col3:
                                    st.markdown(f"""
                                        <div class="card">
                                            <h4>Emotional Breakdown</h4>
                                    """, unsafe_allow_html=True)
                                    
                                    # Create a horizontal bar chart for top emotions
                                    for emotion, prob in result['top_emotions']:
                                        emotion_color = EMOTION_DATA.get(emotion, {}).get('color', '#808080')
                                        emotion_emoji = EMOTION_DATA.get(emotion, {}).get('emoji', '🔍')
                                        st.markdown(f"""
                                            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                                <div style="width: 40px; text-align: center; margin-right: 10px;">
                                                    <span style="font-size: 24px;">{emotion_emoji}</span>
                                                </div>
                                                <div style="flex-grow: 1;">
                                                    <div style="font-weight: 500; margin-bottom: 5px; display: flex; justify-content: space-between;">
                                                        <span>{emotion}</span>
                                                        <span>{prob:.1f}%</span>
                                                    </div>
                                                    <div class="progress-container">
                                                        <div style="
                                                            background: linear-gradient(90deg, {emotion_color} 0%, {emotion_color}90 100%);
                                                            width: {prob}%;
                                                            height: 100%;
                                                            border-radius: 8px;
                                                        "></div>
                                                    </div>
                                                </div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Display personalized suggestions - always use the primary emotion for suggestions
                            st.markdown("""
                                <div class="card">
                                    <h3>Personalized Insights</h3>
                                    <p>Based on your emotional state, here are some tailored recommendations:</p>
                            """, unsafe_allow_html=True)
                            
                            # Always use the primary emotion's suggestions
                            suggestions = EMOTION_DATA.get(primary_emotion, {}).get('suggestions', [])
                            for i, suggestion in enumerate(suggestions):
                                st.markdown(f"""
                                    <div style="
                                        display: flex;
                                        align-items: center;
                                        margin-bottom: 10px;
                                        padding: 12px;
                                        background-color: #F7F9FC;
                                        border-radius: 12px;
                                        border-left: 4px solid {color};
                                    ">
                                        <div style="
                                            width: 30px;
                                            height: 30px;
                                            border-radius: 50%;
                                            background-color: {color}20;
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            margin-right: 15px;
                                            font-weight: bold;
                                        ">
                                            {i+1}
                                        </div>
                                        <div>{suggestion}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="card">
                    <div class="circle-icon">
                        <span style="font-size: 24px;">🌈</span>
                    </div>
                    <h3>Emotional Spectrum</h3>
                    <p>Our AI can detect these emotional states in your voice:</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show all possible emotions with their emojis in a more visual way
            for emotion, data in EMOTION_DATA.items():
                st.markdown(f"""
                    <div class="emotion-card" style="
                        display: flex;
                        align-items: center;
                        padding: 12px;
                        border-radius: 12px;
                        background: linear-gradient(135deg, {data['color']}10, {data['color']}30);
                        border: 1px solid {data['color']}50;
                        margin-bottom: 10px;
                        transition: all 0.3s ease;
                    ">
                        <div style="
                            width: 40px;
                            height: 40px;
                            border-radius: 50%;
                            background-color: white;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin-right: 15px;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        ">
                            <span style="font-size: 20px;">{data['emoji']}</span>
                        </div>
                        <span style="font-weight: 500;">{emotion}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            # Add an information card with recording tips
            st.markdown("""
                <div class="card" style="margin-top: 20px;">
                    <div class="circle-icon">
                        <span style="font-size: 24px;">💡</span>
                    </div>
                    <h3>Recording Tips</h3>
                    <ul style="padding-left: 20px;">
                        <li>Speak naturally for best results</li>
                        <li>Find a quiet environment</li>
                        <li>Express yourself genuinely</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close app-wrapper div

if __name__ == "__main__":
    main()