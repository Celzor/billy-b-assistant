import json
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from . import audio, config
from .logger import logger

# Vosk's small EN model expects 16000 Hz mono audio.
_VOSK_RATE = 16000


class WakeWordListener:
    def __init__(self, on_detected):
        self.on_detected = on_detected
        self._recognizer = None
        self._model = None
        self._stream = None
        self._thread = None
        # Smaller queue: we only need a brief buffer, not 200 stale chunks
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=30)
        self._stop_event = threading.Event()
        self._last_trigger_time = 0.0
        self._wake_phrase = config.LOCAL_WAKE_WORD_PHRASE.lower().strip()
        # Precompute resample ratio (e.g. 48000 -> 16000 = up=1, down=3)
        from math import gcd
        _g = gcd(_VOSK_RATE, audio.MIC_RATE)
        self._resample_up = _VOSK_RATE // _g
        self._resample_down = audio.MIC_RATE // _g

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
            # Always initialise the recogniser at the model's native rate
            recognizer = KaldiRecognizer(model, _VOSK_RATE)
            return model, recognizer
        except Exception as e:
            logger.error(
                f"Failed loading wake-word model at {config.LOCAL_WAKE_WORD_MODEL_PATH}: {e}"
            )
            return None, None

    def _audio_callback(self, indata, frames, time_info, status):
        del frames, time_info
        if status:
            logger.verbose(f"Wake-word status: {status}")
        try:
            self._audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass  # silently drop; overflow logging would create its own lag

    def _to_vosk_rate(self, raw_bytes: bytes) -> bytes:
        """Downsample int16 PCM from MIC_RATE to _VOSK_RATE."""
        if audio.MIC_RATE == _VOSK_RATE:
            return raw_bytes
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        resampled = resample_poly(arr, self._resample_up, self._resample_down)
        return resampled.astype(np.int16).tobytes()

    def _handle_result(self, result_json: str):
        if not result_json:
            return
        try:
            payload = json.loads(result_json)
        except json.JSONDecodeError:
            return
        text = (payload.get("text") or "").lower().strip()
        if not text:
            return
        if self._wake_phrase not in text:
            return

        now = time.time()
        if now - self._last_trigger_time < config.LOCAL_WAKE_WORD_COOLDOWN_SECONDS:
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

        # 8000-sample block at 48000 Hz = ~167 ms per chunk, slow enough
        # to avoid input overflow without lag.
        wake_blocksize = 8000

        try:
            self._stream = sd.RawInputStream(
                samplerate=audio.MIC_RATE,
                blocksize=wake_blocksize,
                device=audio.MIC_DEVICE_INDEX,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            logger.error(f"Failed to start wake-word microphone stream: {e}")
            return

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            vosk_chunk = self._to_vosk_rate(chunk)
            if self._recognizer.AcceptWaveform(vosk_chunk):
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
