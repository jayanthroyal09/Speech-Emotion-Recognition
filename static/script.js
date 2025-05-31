function recordEmotion() {
    console.log("Button clicked. Sending fetch request...");
    const btn = document.getElementById('recordBtn');
    const emotionDiv = document.getElementById('emotion');

    btn.disabled = true;
    btn.textContent = "Recording... Please wait";
    emotionDiv.textContent = "";

    fetch('/predict', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log("Response received:", data);
            const emotion = data.emotion;

            // Emotion-based custom messages
            const messages = {
                "Happy": "😄 Wohoo! Looks like you're feeling happy and cheerful!",
                "Sad": "😢 Oh no! You seem a bit sad. Hope things get better soon!",
                "Angry": "😠 Whoa! You sound a bit angry. Take a deep breath!",
                "Calm": "😌 You seem calm and relaxed. Keep enjoying the peace!",
                "Fearful": "😨 Sounds like you're a little scared. Stay brave!",
                "Surprised": "😲 Ooh! That sounded surprising!",
                "Disgust": "🤢 Hmm… Something seems off. You sound disgusted!"
            };

            // Show friendly message
            emotionDiv.textContent = messages[emotion] || `🧠 Predicted Emotion: ${emotion}`;
            btn.textContent = "Record Again";
            btn.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            emotionDiv.textContent = "❌ Error predicting emotion.";
            btn.textContent = "Try Again";
            btn.disabled = false;
        });
}
