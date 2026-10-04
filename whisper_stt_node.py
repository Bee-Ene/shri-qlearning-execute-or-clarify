"""
Whisper STT Node
ROS2 Humble + macOS M1

Minimum RMS energy threshold, rejects ambient noise and silence
before even attempting transcription. Eliminates hallucinations.
Minimum duration threshold, ignores clips shorter than 0.8s
Expanded hallucination filter word list
Pauses while TTS speaking (/tts_speaking topic)
Routes speech to correct topic based on robot phase:
       READY      → /command_text
       CLARIFYING → /human_feedback
       FEEDBACK   → /human_feedback
Timeout watchdog resets to READY after 45s stuck

Install: pip install openai-whisper sounddevice scipy
         brew install ffmpeg
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import threading
import time


# ── Configuration ─────────────────────────────────────────────────────────────
WHISPER_MODEL     = "small"
SAMPLE_RATE       = 16000
BLOCK_SIZE        = 512

# Energy threshold blocks below this RMS are treated as silence
# Typical speech RMS: 0.02-0.15. Ambient noise: 0.005-0.015.
MIN_RMS_THRESHOLD  = 0.018

SILENCE_RMS        = 0.012    # below this = silence after speech
SILENCE_DURATION   = 1.5      # seconds of silence to end recording
MAX_RECORD_SECS    = 10
MIN_SPEECH_SECS    = 0.8      # ignore clips shorter than this

STATUS_TIMEOUT_SEC = 45.0     # reset to READY if no status update

# Whisper hallucination strings to reject
HALLUCINATIONS = {
    "", ".", "you", "thank you.", "thanks.", "thank you",
    " ", "bye.", "bye", "hmm.", "hmm", "uh", "um", "you.",
    "ah", "oh", "huh", "okay.", "okay", "ok.", "ok",
    "the", "a", "i", "is", "it", "to", "and",
}


class Phase:
    READY      = "ready"
    EXECUTING  = "executing"
    RECOVERING = "recovering"
    CLARIFYING = "clarifying"
    FEEDBACK   = "feedback"


class WhisperSTTNode(Node):

    def __init__(self):
        super().__init__("whisper_stt_node")

        # Publishers
        self.pub_cmd  = self.create_publisher(String, "/command_text",   10)
        self.pub_fb   = self.create_publisher(String, "/human_feedback", 10)

        # Subscriptions
        self.create_subscription(
            String, "/shri_status",          self._status_cb,   10)
        self.create_subscription(
            String, "/tts_speaking",         self._tts_cb,      10)
        self.create_subscription(
            String, "/clarification_query",  self._clar_cb,     10)

        # State
        self._phase          = Phase.READY
        self._tts_busy       = False
        self._tts_done_time  = 0.0
        self._last_status_t  = time.time()
        self._current_q      = ""
        self._listening      = False
        self._lock           = threading.Lock()
        self.TTS_BUFFER      = 2.0   # seconds after TTS finishes before listening

        self.get_logger().info("Loading Whisper model '" + WHISPER_MODEL + "'...")
        self.model = whisper.load_model(WHISPER_MODEL)
        self.get_logger().info("Whisper ready.")
        self.get_logger().info("=" * 55)
        self.get_logger().info("  Whisper STT Node — FINAL VERSION")
        self.get_logger().info("  Speak commands clearly into the microphone.")
        self.get_logger().info("=" * 55)

        threading.Thread(target=self._listen_loop, daemon=True).start()
        threading.Thread(target=self._watchdog,    daemon=True).start()

    # ── Status / TTS callbacks ─────────────────────────────────────────

    def _status_cb(self, msg: String):
        s = msg.data.upper()
        with self._lock:
            self._last_status_t = time.time()
            if   "READY"      in s: self._phase = Phase.READY
            elif "EXECUTING"  in s: self._phase = Phase.EXECUTING
            elif "RECOVERING" in s: self._phase = Phase.RECOVERING
            elif "CLARIFYING" in s: self._phase = Phase.CLARIFYING
            elif "FEEDBACK"   in s: self._phase = Phase.FEEDBACK
            elif "SUCCESS"    in s or "FAILED" in s or "CANCELLED" in s:
                self._phase    = Phase.READY
                self._current_q = ""

    def _tts_cb(self, msg: String):
        with self._lock:
            if msg.data == "speaking":
                self._tts_busy = True
            elif msg.data == "done":
                self._tts_busy     = False
                self._tts_done_time = time.time()

    def _clar_cb(self, msg: String):
        with self._lock:
            self._current_q = msg.data.strip()

    # ── Watchdog ───────────────────────────────────────────────────────

    def _watchdog(self):
        """Force READY if status has not updated for STATUS_TIMEOUT_SEC."""
        while rclpy.ok():
            time.sleep(5.0)
            with self._lock:
                elapsed = time.time() - self._last_status_t
                phase   = self._phase
            if (elapsed > STATUS_TIMEOUT_SEC and
                    phase not in (Phase.READY, Phase.CLARIFYING,
                                  Phase.FEEDBACK)):
                self.get_logger().warn("Watchdog: resetting to READY.")
                with self._lock:
                    self._phase = Phase.READY

    # ── Listen loop ────────────────────────────────────────────────────

    def _listen_loop(self):
        """Continuously listens and routes speech to correct topic."""
        while rclpy.ok():
            with self._lock:
                phase        = self._phase
                tts_busy     = self._tts_busy
                tts_done_t   = self._tts_done_time
                current_q    = self._current_q

            # Do not listen while TTS is speaking or in buffer window
            if tts_busy or (time.time() - tts_done_t < self.TTS_BUFFER):
                time.sleep(0.2)
                continue

            # Do not listen while robot is physically busy
            if phase in (Phase.EXECUTING, Phase.RECOVERING):
                time.sleep(0.3)
                continue

            # Print listening prompt
            if phase == Phase.READY:
                self.get_logger().info(
                    "\n  [LISTENING] Speak your command...")
            elif phase == Phase.CLARIFYING:
                q_short = (current_q[:50] + "..."
                           if len(current_q) > 50 else current_q)
                self.get_logger().info(
                    "\n  [LISTENING] Robot asked: " + q_short +
                    "\n  Speak your answer.")
            elif phase == Phase.FEEDBACK:
                self.get_logger().info(
                    "\n  [LISTENING] Say YES or NO.")

            # Record audio
            audio = self._record()
            if audio is None:
                time.sleep(0.1)
                continue

            # Duration check
            duration = len(audio) / SAMPLE_RATE
            if duration < MIN_SPEECH_SECS:
                continue

            # Energy check reject clips that are mostly silence
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < MIN_RMS_THRESHOLD:
                self.get_logger().info(
                    "  (Audio too quiet, skipping. RMS=" +
                    str(round(rms, 4)) + ")")
                continue

            # Transcribe
            text = self._transcribe(audio)
            if not text:
                self.get_logger().info("  (No transcription)")
                continue

            self.get_logger().info("  Transcribed: '" + text + "'")

            # Route to correct topic
            with self._lock:
                current_phase = self._phase
                # Block listening immediately
                self._phase = Phase.EXECUTING

            out = String()
            out.data = text

            if current_phase == Phase.READY:
                self.pub_cmd.publish(out)
                self.get_logger().info(
                    "  → /command_text: '" + text + "'")
            else:
                self.pub_fb.publish(out)
                self.get_logger().info(
                    "  → /human_feedback: '" + text + "'")

            # Short pause to avoid double-capture
            time.sleep(1.0)

    # ── Audio recording ────────────────────────────────────────────────

    def _record(self) -> np.ndarray:
        """Record until silence or max duration. Returns float32 array."""
        with self._lock:
            if self._listening:
                return None
            self._listening = True

        frames        = []
        silence_count = 0
        silence_limit = int(SAMPLE_RATE * SILENCE_DURATION / BLOCK_SIZE)
        max_frames    = int(SAMPLE_RATE * MAX_RECORD_SECS  / BLOCK_SIZE)
        started       = False

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE,
                                 channels=1,
                                 dtype="float32",
                                 blocksize=BLOCK_SIZE) as stream:
                for _ in range(max_frames):
                    block, _ = stream.read(BLOCK_SIZE)
                    rms = float(np.sqrt(np.mean(block ** 2)))
                    frames.append(block.copy())

                    if rms > SILENCE_RMS:
                        started       = True
                        silence_count = 0
                    elif started:
                        silence_count += 1
                        if silence_count >= silence_limit:
                            break

        except Exception as e:
            self.get_logger().error("Microphone error: " + str(e))
            with self._lock:
                self._listening = False
            return None

        with self._lock:
            self._listening = False

        if not started or not frames:
            return None

        return np.concatenate(frames, axis=0).flatten()

    # ── Transcription ──────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run Whisper on audio array. Returns cleaned text or empty string."""
        try:
            with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False) as f:
                tmp = f.name
                wav.write(tmp, SAMPLE_RATE,
                          (audio * 32767).astype(np.int16))

            result = self.model.transcribe(
                tmp,
                language="en",
                fp16=False,       # M1 CPU does not support fp16
                verbose=False,
                # Suppress non-speech tokens more aggressively
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            os.unlink(tmp)

            text = result["text"].strip().lower()
            text = text.strip(".,!?;:'\"")

            if text in HALLUCINATIONS:
                return ""
            if len(text) < 3:
                return ""

            return text

        except Exception as e:
            self.get_logger().error("Transcription error: " + str(e))
            return ""


def main(args=None):
    rclpy.init(args=args)
    node = WhisperSTTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
