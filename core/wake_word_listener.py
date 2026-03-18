import json
import os
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
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._restart_stream_event = threading.Event()
        self._last_trigger_time = 0.0
        self._wake_phrase = config.LOCAL_WAKE_WORD_PHRASE.lower().strip()
        self._last_status_log_time = 0.0
        self._status_log_interval_seconds = 5.0
        self._recent_phrases: list[str] = []
        self._last_overflow_log_time = 0.0
        self._overflow_log_interval_seconds = 5.0
        self._overflow_count = 0
        self._samplerate = self._pick_samplerate()
        self._blocksize = max(1, int(self._samplerate * 0.25))
        self._max_blocksize = max(self._blocksize, self._samplerate)

    def _pick_samplerate(self) -> int:
        preferred_rates = [16000]
        if audio.MIC_RATE not in preferred_rates:
            preferred_rates.append(audio.MIC_RATE)
        preferred_rates.extend(
            rate for rate in [24000, 48000, 44100] if rate not in preferred_rates
        )

        for rate in preferred_rates:
            try:
                sd.check_input_settings(
                    device=audio.MIC_DEVICE_INDEX,
                    samplerate=rate,
                    channels=1,
                    dtype='int16',
                )
                return rate
            except Exception:
                continue

        logger.warning(
            f"Wake-word listener could not validate a mono input rate; falling back to MIC_RATE={audio.MIC_RATE}",
            "⚠️",
        )
        return audio.MIC_RATE

    def _ensure_model(self):
        try:
            from vosk import KaldiRecognizer, Model
        except ImportError:
            logger.warning(
                "LOCAL_WAKE_WORD_ENABLED is true but 'vosk' is not installed", "⚠️"
            )
            return None, None

        model_path = config.LOCAL_WAKE_WORD_MODEL_PATH
        if not os.path.isdir(model_path):
            logger.error(
                "Wake-word model directory is missing: "
                f"{model_path}. Download a Vosk model on the Raspberry Pi and set "
                "LOCAL_WAKE_WORD_MODEL_PATH to that folder.",
            )
            return None, None

        try:
            model = Model(model_path)
            recognizer = KaldiRecognizer(model, self._samplerate)
            return model, recognizer
        except Exception as e:
            logger.error(
                f"Failed loading wake-word model at {config.LOCAL_WAKE_WORD_MODEL_PATH}: {e}"
            )
            return None, None

    def _request_stream_restart(self, reason: str):
        if self._restart_stream_event.is_set():
            return

        old_blocksize = self._blocksize
        self._blocksize = min(self._blocksize * 2, self._max_blocksize)
        if self._blocksize == old_blocksize:
            return

        logger.warning(
            f"Restarting wake-word stream after {reason}; increasing blocksize from {old_blocksize} to {self._blocksize}",
            "🔁",
        )
        self._restart_stream_event.set()

    def _audio_callback(self, indata, frames, time_info, status):
        del frames, time_info
        if status:
            now = time.time()
            self._overflow_count += 1
            if (
                now - self._last_overflow_log_time
                >= self._overflow_log_interval_seconds
            ):
                logger.warning(
                    "Wake-word audio status: "
                    f"{status} | samplerate={self._samplerate} | blocksize={self._blocksize}",
                    "⚠️",
                )
                self._last_overflow_log_time = now

            if self._overflow_count >= 3:
                self._request_stream_restart(str(status))
                self._overflow_count = 0
        try:
            self._audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            logger.warning("Wake-word queue full; dropping chunk", "⚠️")
            self._request_stream_restart("queue saturation")

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
            f"target='{config.LOCAL_WAKE_WORD_PHRASE}' | samplerate={self._samplerate} | "
            f"blocksize={self._blocksize} | queue={self._audio_queue.qsize()} | "
            f"heard_recent={heard_text}",
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

    def _open_stream(self):
        self._stream = sd.RawInputStream(
            samplerate=self._samplerate,
            blocksize=self._blocksize,
            device=audio.MIC_DEVICE_INDEX,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
            latency="high",
        )
        self._stream.start()
        self._overflow_count = 0
        logger.info(
            f"Wake-word microphone stream started | samplerate={self._samplerate} | blocksize={self._blocksize}",
            "🎙️",
        )

    def _close_stream(self):
        if not self._stream:
            return

        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stream = None

    def _restart_stream_if_requested(self):
        if not self._restart_stream_event.is_set():
            return

        self._restart_stream_event.clear()
        self._close_stream()
        with self._audio_queue.mutex:
            self._audio_queue.queue.clear()

        try:
            self._open_stream()
        except Exception as e:
            logger.error(f"Failed to restart wake-word microphone stream: {e}")
            self._stop_event.set()

    def _run(self):
        self._model, self._recognizer = self._ensure_model()
        if self._recognizer is None:
            return

        logger.info(
            f"Wake-word listener active for phrase: '{config.LOCAL_WAKE_WORD_PHRASE}'",
            "🛎️",
        )

        try:
            self._open_stream()
        except Exception as e:
            logger.error(f"Failed to start wake-word microphone stream: {e}")
            return

        while not self._stop_event.is_set():
            self._restart_stream_if_requested()
            self._maybe_log_status()

            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._recognizer.AcceptWaveform(chunk):
                self._handle_result(self._recognizer.Result())
            else:
                self._handle_result(self._recognizer.PartialResult())

        self._close_stream()
        logger.info("Wake-word listener stopped", "🛑")

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
