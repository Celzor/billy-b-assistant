import json
import queue
import threading
import time

import numpy as np
import sounddevice as sd

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

    def _downsample(self, raw_bytes: bytes, from_rate: int) -> bytes:
        """Fast integer decimation to _VOSK_RATE via numpy slice.

        Simple striding (arr[::step]) is orders of magnitude faster than
        resample_poly on slow hardware. Aliasing above _VOSK_RATE/2 is
        acceptable for wake-word detection (speech sits below ~4 kHz).
        """
        step = from_rate // _VOSK_RATE
        if step <= 1:
            return raw_bytes
        arr = np.frombuffer(raw_bytes, dtype=np.int16)
        return arr[::step].tobytes()

    def _handle_result(self, result_json: str) -> bool:
        """Returns True if the wake phrase was detected, False otherwise."""
        if not result_json:
            return False
        try:
            payload = json.loads(result_json)
        except json.JSONDecodeError:
            return False
        text = (payload.get("text") or "").lower().strip()
        if not text:
            return False
        if self._wake_phrase not in text:
            return False

        now = time.time()
        if now - self._last_trigger_time < config.LOCAL_WAKE_WORD_COOLDOWN_SECONDS:
            return False
        self._last_trigger_time = now

        logger.success(f"Wake phrase detected: '{text}'", "👂")
        return True

    def _run(self):
        self._model, self._recognizer = self._ensure_model()
        if self._recognizer is None:
            return

        logger.info(
            f"Wake-word listener active for phrase: '{config.LOCAL_WAKE_WORD_PHRASE}'",
            "🛎️",
        )

        # Try to open the mic directly at the model's native rate so no
        # resampling is needed at all.  If the hardware doesn't support 16 kHz,
        # fall back to the system mic rate and downsample via fast integer
        # decimation (arr[::step], not resample_poly).
        stream_rate = None
        for candidate in (_VOSK_RATE, audio.MIC_RATE):
            try:
                sd.check_input_settings(
                    device=audio.MIC_DEVICE_INDEX, samplerate=candidate, channels=1
                )
                stream_rate = candidate
                break
            except Exception:
                continue

        if stream_rate is None:
            logger.error("Wake-word mic supports neither 16000 Hz nor the system mic rate")
            return

        if stream_rate == _VOSK_RATE:
            logger.info(f"Wake-word stream at {stream_rate} Hz — no resampling needed", "🎙️")
        else:
            logger.info(
                f"Wake-word stream at {stream_rate} Hz — decimating to {_VOSK_RATE} Hz", "🎙️"
            )

        # At 16000 Hz, 8000 samples = 500 ms per chunk (2 callbacks/s).
        # At 48000 Hz, 8000 samples = 167 ms per chunk (6 callbacks/s).
        # Both are slow enough that processing never falls behind.
        wake_blocksize = 8000
        detected = False

        try:
            self._stream = sd.RawInputStream(
                samplerate=stream_rate,
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

            vosk_chunk = self._downsample(chunk, stream_rate) if stream_rate != _VOSK_RATE else chunk
            if self._recognizer.AcceptWaveform(vosk_chunk):
                if self._handle_result(self._recognizer.Result()):
                    detected = True
                    self._stop_event.set()
            else:
                if self._handle_result(self._recognizer.PartialResult()):
                    detected = True
                    self._stop_event.set()

        # Release the ALSA device BEFORE notifying the session.
        # If on_detected() is called while the stream is still open, ALSA will
        # refuse the session's attempt to open the same mic device.
        try:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        except Exception:
            pass

        logger.info("Wake-word listener stopped", "🛑")

        if detected:
            self.on_detected()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
