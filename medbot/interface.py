import gradio as gr
import threading
import re
import requests
import sounddevice as sd
import soundfile as sf
import tempfile
import io
import numpy as np 
from scipy.io.wavfile import write as wav_write

from .handlers import respond, reset_chat
from .config import NGROK_URL, VOICE_SAMPLE_RATE, VOICE_CHANNELS


class VoiceEnhancedInterface:
    def __init__(self):
        # Load from config
        self.NGROK_URL = NGROK_URL
        if not self.NGROK_URL:
            print("Warning: NGROK_URL is not set in config/environment.")
            self.NGROK_URL = "https://561ce706eda5.ngrok-free.app/" # Fallback
            
        self.is_bot_speaking = False         # True when TTS playback is active
        self.is_recording = False            # True while mic is recording
        self._record_thread = None
        self._last_transcript = ""           # populated after recording stops
        self.samplerate = VOICE_SAMPLE_RATE  # Load from config
        self.conversation_mode = False

    # --- Message Formatting (unchanged) ---
    def _format_message(self, text):
        """Formats text into a two-row table, fixing the extra top padding."""
        return f"""<table style="width: 100%; border: none; border-collapse: collapse;">
<tr style="border: none;">
<td style="border: none; text-align: right; padding: 0 4px 2px 0; font-size: 1.2em; cursor: pointer;" title="Click message to hear audio">🔊</td>
</tr>
<tr style="border: none;">
<td class="message-text" style="border: none; padding: 0; white-space: pre-wrap;">{text}</td>
</tr>
</table>"""

    def _extract_text(self, formatted_text):
        """Extracts the original raw text from our HTML structure."""
        if not isinstance(formatted_text, str):
            return str(formatted_text)
        match = re.search(r'<td class="message-text"[^>]*>([\s\S]*?)</td>', formatted_text)
        return match.group(1).strip() if match else formatted_text

    # --- STT via ngrok ---
    def transcribe_audio(self, wav_path):
        """Send recorded audio to /transcribe endpoint via ngrok."""
        try:
            with open(wav_path, "rb") as f:
                resp = requests.post(f"{self.NGROK_URL}/transcribe", files={"file": f}, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("text", "")
            else:
                print("Transcribe failed:", resp.status_code, resp.text)
                return ""
        except Exception as e:
            print("STT Error:", e)
            return ""

    # --- TTS via ngrok (fetch mp3, play via sounddevice) ---
    def _fetch_tts_bytes(self, text):
        """Fetch MP3 bytes from the TTS endpoint and convert to numpy audio."""
        try:
            resp = requests.post(f"{self.NGROK_URL}/speak", data={"text": text}, timeout=60)
            if resp.status_code == 200:
                return io.BytesIO(resp.content)
            else:
                print("TTS fetch failed:", resp.status_code, resp.text)
                return None
        except Exception as e:
            print("TTS request error:", e)
            return None
        
    def speak_text(self, text):
        """Toggleable TTS playback with smooth streaming audio."""
        try:
            # --- Stop if already speaking ---
            if self.is_bot_speaking:
                print("🛑 Stopping TTS playback...")
                self.is_bot_speaking = False
                sd.stop()
                return

            # --- Fetch audio bytes ---
            tts_bytes = self._fetch_tts_bytes(text)
            if not tts_bytes:
                return

            # --- Read MP3 into float32 array ---
            data, sr = sf.read(tts_bytes, dtype="float32")

            # --- Begin playback in background thread ---
            self.is_bot_speaking = True
            print("🔊 Starting smooth TTS playback...")

            def play_worker():
                try:
                    with sd.OutputStream(samplerate=sr, channels=data.shape[1] if len(data.shape) > 1 else 1) as stream:
                        block_size = 4096
                        i = 0
                        while self.is_bot_speaking and i < len(data):
                            end = min(i + block_size, len(data))
                            stream.write(data[i:end])
                            i = end
                        stream.stop()
                except Exception as e:
                    print("Playback error:", e)
                finally:
                    self.is_bot_speaking = False
                    print("✅ TTS playback finished.")

            threading.Thread(target=play_worker, daemon=True).start()

        except Exception as e:
            print("TTS Error:", e)
            self.is_bot_speaking = False
            sd.stop()


    # --- Chat Logic (same behavior, removed auto-speak) ---
    def user_message_handler(self, user_message, chat_history):
        if not user_message.strip():
            return "", chat_history
        # interrupt TTS if user sends new text
        if self.is_bot_speaking:
            sd.stop()
            self.is_bot_speaking = False

        formatted_user_msg = self._format_message(user_message)
        chat_history.append((formatted_user_msg, None))
        return "", chat_history

    def bot_response_handler(self, chat_history):
        if not chat_history or chat_history[-1][1] is not None:
            return chat_history

        formatted_user_msg = chat_history[-1][0]
        raw_user_msg = self._extract_text(formatted_user_msg)

        raw_history = [
            (self._extract_text(q), self._extract_text(a))
            for q, a in chat_history[:-1] if q and a
        ]

        _, updated_raw_history = respond(raw_user_msg, raw_history)
        raw_bot_response = updated_raw_history[-1][1]

        formatted_bot_response = self._format_message(raw_bot_response)
        chat_history[-1] = (formatted_user_msg, formatted_bot_response)

        # 🔊 Auto-speak if conversation mode is ON
        if self.conversation_mode:
            print("🗣️ Conversation mode active — auto-speaking bot response.")
            # stop previous speech if still ongoing
            if self.is_bot_speaking:
                sd.stop()
                self.is_bot_speaking = False
            # run speaking in background
            threading.Thread(
                target=self.speak_text,
                args=(raw_bot_response,),
                daemon=True
            ).start()

        return chat_history


    # ---------------------------
    # RECORDING: start / stop
    # ---------------------------
    def handle_voice_input(self, chat_history):
        """
        This function is wired to the mic button.
        - First click: starts recording in background and returns a "Recording..." message.
        - Second click: signals the recording thread to stop, waits for it to finish,
                       then returns the transcribed text appended to chat_history.
        Returns a tuple matching Gradio outputs wired in your app: (msg_value, chat_history)
        """
        if not self.is_recording:
            # start background recording thread
            self.is_recording = True
            self._last_transcript = ""
            self._record_thread = threading.Thread(target=self._record_worker, daemon=True)
            self._record_thread.start()
            # return a message visible in the text input (you used msg as output before)
            return "🎙️ Recording... Click mic again to stop.", chat_history
        else:
            # stop recording and wait for worker to finish
            self.is_recording = False
            if self._record_thread is not None:
                self._record_thread.join(timeout=30)  # avoid infinite hang
            # if we got a transcript, treat it like user input
            transcript = getattr(self, "_last_transcript", "") or ""
            if transcript:
                # append user message and then generate bot reply (like in old flow)
                return self.user_message_handler(transcript, chat_history)
            else:
                # no clear transcript
                chat_history.append(("Voice input error", "Sorry, I couldn't hear clearly."))
                return "", chat_history

    def _record_worker(self):
        """Background worker that records until self.is_recording becomes False,
           then writes WAV, calls transcribe_audio, and stores transcript in self._last_transcript.
        """
        frames = []
        try:
            # Use blocking InputStream with callback to collect frames
            def callback(indata, frames_count, time_info, status):
                # append copy of indata (numPy array)
                frames.append(indata.copy())

            with sd.InputStream(samplerate=self.samplerate, channels=VOICE_CHANNELS, callback=callback):
                # spin until user stops recording
                while self.is_recording:
                    sd.sleep(100)
        except Exception as e:
            print("Recording error:", e)

        # combine frames into a single numpy array (shape: samples x channels)
        if len(frames) == 0:
            self._last_transcript = ""
            return
        try:
            audio_np = np.concatenate(frames, axis=0)
            # write to temporary wav file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                wav_write(tmp.name, self.samplerate, audio_np)
                # send to transcribe
                text = self.transcribe_audio(tmp.name)
                self._last_transcript = text or ""
        except Exception as e:
            print("Error processing recorded audio:", e)
            self._last_transcript = ""

    # ---------------------------
    # Click-to-speak per message
    # ---------------------------
    def speak_selected_message(self, evt: gr.SelectData):
        """
        This is triggered when a user clicks any message in the Chatbot.
        It extracts the message text and toggles playback for that message.
        """
        try:
            selected_formatted_text = evt.value
            text_to_speak = self._extract_text(selected_formatted_text)
            if not text_to_speak:
                return
            # run TTS in background to avoid blocking UI
            threading.Thread(target=self.speak_text, args=(text_to_speak,), daemon=True).start()
        except Exception as e:
            print("speak_selected_message error:", e)

    # ---------------------------
    # Conversation mode toggle (keeps same UI behavior)
    # ---------------------------
    def toggle_conversation_mode(self):
        self.conversation_mode = not self.conversation_mode
        return "🎙️ Conversation Mode: ON" if self.conversation_mode else "⏸️ Conversation Mode: OFF"

    # ---------------------------
    # Build the Gradio UI (preserves old layout)
    # ---------------------------
    def build_interface(self):
        with gr.Blocks(
            theme="soft",
            title="BluePlanet Medical Assistant",
            css="""
            .voice-button { background: linear-gradient(45deg, #d7f2f5, #db2127) !important; color: white !important; border: none !important; border-radius: 50% !important; width: 50px !important; height: 50px !important; font-size: 20px !important; }
            .conversation-circle-button { background: linear-gradient(45deg, #d7f2f5, #21db8b) !important; color: white !important; border: none !important; border-radius: 50% !important; width: 50px !important; height: 50px !important; font-size: 20px !important; }
            """
        ) as demo:
            gr.Markdown("# 🏥 BluePlanet Medical Assistant")
            gr.Markdown("**By BluePlanet Solutions AI - Powered by Whisper + gTTS (via ngrok)**")

            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        height=500, elem_id="medical-chatbot",
                        show_copy_button=True, bubble_full_width=False, render_markdown=True
                    )
                    with gr.Row():
                        with gr.Column(scale=8):
                            msg = gr.Textbox(
                                placeholder="Tell me about your symptoms or health concerns... (or use voice input)",
                                label="Your Message", lines=2, max_lines=4
                            )
                        with gr.Column(scale=1, min_width=60):
                            voice_input_btn = gr.Button("🎤", elem_classes=["voice-button"], size="sm")
                            conversation_mode_btn = gr.Button("🗣️", elem_classes=["conversation-circle-button"], size="sm")
                    with gr.Row():
                        send_btn = gr.Button("📤 Send", variant="primary", scale=2)

                with gr.Column(scale=1):
                    reset_btn = gr.Button("🔄 Start New Consultation", variant="secondary", size="lg")
                    gr.Markdown("### 🎙️ Voice Controls")
                    gr.Markdown("""
                    **🔊 Click Any Message**: Reads the text aloud.  
                    **🎤 Voice Input**: Click mic to speak your symptoms.  
                    **🗣️ Conversation**: Hands-free conversation.  
                    **⚡ Quick Actions**: Interrupt bot by speaking.
                    """)
                    gr.Markdown("### 🧠 Memory Features")
                    gr.Markdown("""
                    - 📋 Tracks symptoms & timeline  
                    - 💊 Remembers medications & allergies  
                    - 🔄 Maintains conversation context  
                    - 📊 Provides comprehensive summaries
                    """)
                    conversation_status = gr.Textbox(value="⏸️ Conversation Mode: OFF", label="Mode Status", interactive=False)

            gr.Examples(
                examples=[
                    "I have a persistent cough and sore throat for 3 days",
                    "I've been having severe headaches and feel dizzy",
                    "My stomach hurts and I feel nauseous after eating",
                ],
                inputs=msg,
                label="💡 Try these example symptoms:"
            )

            # Hookups (preserve original wiring)
            msg.submit(self.user_message_handler, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot_response_handler, chatbot, chatbot
            )
            send_btn.click(self.user_message_handler, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot_response_handler, chatbot, chatbot
            )

            # voice button: toggles recording. it returns (msg_text, chat_history)
            voice_input_btn.click(self.handle_voice_input, [chatbot], [msg, chatbot], queue=False).then(
                self.bot_response_handler, chatbot, chatbot
            )

            conversation_mode_btn.click(self.toggle_conversation_mode, [], conversation_status)
            reset_btn.click(reset_chat, [], [chatbot, msg])

            # clicking a chat message triggers speak_selected_message
            chatbot.select(self.speak_selected_message, None, None)

        return demo


# --- Run Interface ---
interface_instance = VoiceEnhancedInterface()


def build_interface():

    return interface_instance.build_interface()
