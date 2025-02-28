from openai import OpenAI
import gradio as gr
from gtts import gTTS
import os
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def translate_audio(audio_file_path):
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "Error: No valid audio file uploaded.", "", ""

    with open(audio_file_path, "rb") as audio:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    english_text = transcript.text.strip()
    
    prompt = f"Translate the following English text to Yoruba:\n\n{english_text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a translation assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    yoruba_text = response.choices[0].message.content.strip()
    
    return english_text, yoruba_text, None  # No Yoruba audio output

# Update Gradio Interface to Remove Yoruba Audio Output
iface = gr.Interface(
    fn=translate_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs=[
        gr.Textbox(label="English Transcription"),
        gr.Textbox(label="Yoruba Translation")
    ],
    title="Real-Time English-to-Yoruba Translator",
    description="Speak in English and get the translation in Yoruba (text only)."
)


if __name__ == "__main__":
    iface.launch()
