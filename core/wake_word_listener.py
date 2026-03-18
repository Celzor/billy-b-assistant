import json
import queue
import threading
import time

import sounddevice as sd

from . import audio, config
from .logger import logger


class WakeWordListener:
    def __init__(self, on_detected):
        self.on_detected = on_detected
        self._recognizer = None
        self._model = None
        self._stream = None
        self._thread = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()
        self._last_trigger_time = 0.0
        self._wake_phrase = config.LOCAL_WAKE_WORD_PHRASE.lower().strip()
        self._last_status_log_time = 0.0
        self._status_log_interval_seconds = 5.0
        self._recent_phrases: list[str] = []

    def _ensure_model(self):
        try:
            from vosk import KaldiRecognizer, Model
        except ImportError:
            logger.warning(
                "LOCAL_WAKE_WORD_ENABLED is true but 'vosk' is not installed", "⚠️"
            )
            return None, None

        try:
            model = Model(config.LOCAL_WAKE_WORD_MODEL_PATH)
            recognizer = KaldiRecognizer(model, audio.MIC_RATE)
            return model, recognizer
        except Exception as e:
            logger.error(
                f"Failed loading wake-word model at {config.LOCAL_WAKE_WORD_MODEL_PATH}: {e}"
            )
            return None, None

    def _audio_callback(self, indata, frames, time_info, status):
        del frames, time_info
        if status:
            logger.verbose(f"Wake-word audio status: {status}")
        try:
            self._audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            logger.warning("Wake-word queue full; dropping chunk", "⚠️")

    def _track_phrase(self, text: str):
        if not text:
            return

        if self._recent_phrases and self._recent_phrases[-1] == text:
            return

        self._recent_phrases.append(text)

        if len(self._recent_phrases) > 10:
            self._recent_phrases = self._recent_phrases[-10:]

    def _maybe_log_status(self):
        now = time.time()
        if now - self._last_status_log_time < self._status_log_interval_seconds:
            return

        heard_text = ", ".join(self._recent_phrases) if self._recent_phrases else "none"
        logger.info(
            "Wake-word listener running | "
            f"target='{config.LOCAL_WAKE_WORD_PHRASE}' | heard_recent={heard_text}",
            "🩺",
        )
        self._recent_phrases.clear()
        self._last_status_log_time = now

    def _handle_result(self, result_json: str):
        if not result_json:
            return

        try:
            payload = json.loads(result_json)
        except json.JSONDecodeError:
            return

        text = (payload.get("text") or payload.get("partial") or "").lower().strip()
        if not text:
            return

        self._track_phrase(text)

        if self._wake_phrase not in text:
            return

        now = time.time()
        if now - self._last_trigger_time < config.LOCAL_WAKE_WORD_COOLDOWN_SECONDS:
            logger.info(
                f"Wake phrase heard during cooldown ({config.LOCAL_WAKE_WORD_COOLDOWN_SECONDS}s): '{text}'",
                "⏱️",
            )
            return
        self._last_trigger_time = now

        logger.success(f"Wake phrase detected: '{text}'", "👂")
        self.on_detected()

    def _run(self):
        self._model, self._recognizer = self._ensure_model()
        if self._recognizer is None:
            return

        logger.info(
            f"Wake-word listener active for phrase: '{config.LOCAL_WAKE_WORD_PHRASE}'",
            "🛎️",
        )

        try:
            self._stream = sd.RawInputStream(
                samplerate=audio.MIC_RATE,
                blocksize=audio.CHUNK_SIZE,
                device=audio.MIC_DEVICE_INDEX,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Wake-word microphone stream started", "🎙️")
        except Exception as e:
            logger.error(f"Failed to start wake-word microphone stream: {e}")
            return

        while not self._stop_event.is_set():
            self._maybe_log_status()

            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._recognizer.AcceptWaveform(chunk):
                self._handle_result(self._recognizer.Result())
            else:
                self._handle_result(self._recognizer.PartialResult())

        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

        logger.info("Wake-word listener stopped", "🛑")

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
