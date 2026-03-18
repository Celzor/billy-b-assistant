import asyncio
import contextlib
import threading
import time
from concurrent.futures import CancelledError

from . import audio, config
from .logger import logger
from .movements import move_head, move_tail


try:
    from gpiozero import Button
    from gpiozero.mixins import HoldThread

    gpiozero_available = True
except ImportError:
    gpiozero_available = False

if config.MOCKFISH or not gpiozero_available:
    # Mock button in mockfish mode or if gpiozero not available
    class MockButton:
        def __init__(self, pin, pull_up=True):
            self.pin = pin
            self.when_pressed = None
            self.is_pressed = False
            if config.MOCKFISH:
                logger.info(f"Mockfish: Button on pin {pin} mocked", "🐟")
            elif not gpiozero_available:
                logger.info(f"gpiozero not available: Button on pin {pin} mocked", "🐟")
                # Start thread to listen for Enter key using pynput
                import threading

                threading.Thread(target=self._listen, daemon=True).start()

        def _listen(self):
            # Use fallback input() method for mock button
            self._fallback_listen()

        def _fallback_listen(self):
            import sys

            if not sys.stdin.isatty():
                logger.warning(
                    "stdin is not a tty, mock button input not available", "⚠️"
                )
                return
            while True:
                try:
                    user_input = input("Press Enter to simulate button press: ")
                    if user_input == "" and self.when_pressed:
                        self.when_pressed()
                except (EOFError, KeyboardInterrupt):
                    break

        def close(self):
            pass

    Button = MockButton
from .movements import move_head
from .session_manager import BillySession
from .wake_word_listener import WakeWordListener


# Button and session globals
is_active = False
session_thread = None
interrupt_event = threading.Event()
session_instance: BillySession | None = None
last_button_time = 0
button_debounce_delay = 0.5  # seconds debounce
_session_start_lock = threading.Lock()  # Lock to prevent concurrent session starts

# Setup hardware button
button = Button(config.BUTTON_PIN, pull_up=True)
wake_word_listener = None


def _force_release_session_start_lock(reason: str):
    """Best-effort lock release to recover from stuck session threads."""
    if _session_start_lock.locked():
        try:
            _session_start_lock.release()
            logger.warning(f"Force-released session start lock ({reason})", "🧹")
        except RuntimeError:
            # Lock may have been released concurrently.
            pass


def _ensure_button_hold_thread():
    """Work around rare gpiozero race where _hold_thread becomes None."""
    if config.MOCKFISH or not gpiozero_available:
        return
    try:
        hold_thread = getattr(button, "_hold_thread", None)
        if hold_thread is None:
            button._hold_thread = HoldThread(button)
            logger.warning("Repaired gpiozero button hold thread", "🛠️")
    except Exception as e:
        logger.warning(f"Failed to repair button hold thread: {e}", "⚠️")


def is_billy_speaking():
    """Return True if Billy is playing audio (wake-up or response)."""
    if not audio.playback_done_event.is_set():
        return True
    return bool(not audio.playback_queue.empty())


def trigger_session(source: str = "button", require_physical_press: bool = False):
    global \
        is_active, \
        session_thread, \
        interrupt_event, \
        session_instance, \
        last_button_time

    now = time.time()
    if now - last_button_time < button_debounce_delay:
        return  # Ignore very quick repeat presses (debounce)
    last_button_time = now

    if require_physical_press and not button.is_pressed:
        return

    if is_active:
        logger.info(f"{source.capitalize()} trigger during active session.", "🔁")
        if (
            session_instance
            and session_instance.loop
            and session_instance.is_assistant_turn()
        ):
            try:
                logger.info("Assistant is speaking. Handing turn back to user...", "🎙️")
                future = asyncio.run_coroutine_threadsafe(
                    session_instance.interrupt_to_user_turn(), session_instance.loop
                )
                future.result(timeout=3.0)
                logger.success("Turn handed back to user (mic open).")
                return
            except Exception as e:
                logger.warning(f"Turn handoff failed, stopping session instead: {e}")

        interrupt_event.set()
        audio.stop_playback()

        if session_instance:
            try:
                logger.info("Stopping active session...", "🛑")
                # A concurrent.futures.CancelledError is expected here, because the last
                # thing that BillySession.stop_session does is `await asyncio.sleep`,
                # and that will raise CancelledError because it's a logical place to
                # stop.
                with contextlib.suppress(CancelledError):
                    future = asyncio.run_coroutine_threadsafe(
                        session_instance.stop_session(), session_instance.loop
                    )
                    # Add timeout to prevent hanging
                    try:
                        future.result(timeout=5.0)  # Wait up to 5 seconds
                        logger.success("Session stopped.")
                    except TimeoutError:
                        logger.warning("Session stop timeout, forcing cleanup")
                        with contextlib.suppress(Exception):
                            force_stop_future = asyncio.run_coroutine_threadsafe(
                                session_instance.request_stop(), session_instance.loop
                            )
                            force_stop_future.result(timeout=1.0)
                        with contextlib.suppress(Exception):
                            force_close_future = asyncio.run_coroutine_threadsafe(
                                session_instance._close_ws(timeout=0.5),
                                session_instance.loop,
                            )
                            force_close_future.result(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error stopping session ({type(e)}): {e}")
            finally:
                # Always ensure cleanup
                session_instance = None
                # Wait for session thread to finish to ensure mic is fully closed
                if session_thread and session_thread.is_alive():
                    logger.info("Waiting for session thread to finish...", "⏳")
                    session_thread.join(timeout=2.0)
                    if session_thread.is_alive():
                        logger.warning("Session thread did not finish in time", "⚠️")
                        # Do not keep the start lock blocked forever by a stuck thread.
                        _force_release_session_start_lock("session thread timeout")
        is_active = False  # ✅ Ensure this is always set after stopping
        return

    # Use lock to prevent concurrent session starts (but allow interruption above)
    if not _session_start_lock.acquire(blocking=False):
        # Recovery path: if no active session is running, try to clear a stale lock/thread.
        if not is_active:
            logger.warning(
                "Session start lock busy while inactive; attempting recovery", "🧯"
            )
            if session_thread and session_thread.is_alive():
                interrupt_event.set()
                audio.stop_playback()
                if session_instance and session_instance.loop:
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            session_instance.stop_session(), session_instance.loop
                        )
                        with contextlib.suppress(Exception):
                            future.result(timeout=1.5)
                    except Exception as e:
                        logger.warning(
                            f"Stale-session stop during recovery failed: {e}", "⚠️"
                        )
                session_thread.join(timeout=1.0)
            if session_thread and session_thread.is_alive():
                logger.warning(
                    "Session thread still alive during recovery; forcing stale cleanup",
                    "⚠️",
                )
            # Always attempt to recover lock here; stale thread may linger but should not
            # block new button presses indefinitely.
            session_instance = None
            session_thread = None
            _force_release_session_start_lock("inactive stale lock recovery")
            if _session_start_lock.acquire(blocking=False):
                logger.info("Recovered stale session lock; continuing start", "✅")
            else:
                logger.warning("Could not recover session lock yet, try again", "⚠️")
                return
        else:
            logger.warning(
                "Session start already in progress, ignoring button press", "⚠️"
            )
            return

    try:
        # Ensure previous session thread is fully finished before starting new one
        if session_thread and session_thread.is_alive():
            logger.warning("Previous session thread still running, waiting...", "⏳")
            session_thread.join(timeout=2.0)
            if session_thread.is_alive():
                logger.warning(
                    "Previous session thread did not finish, attempting forced stop",
                    "⚠️",
                )
                if session_instance and session_instance.loop:
                    with contextlib.suppress(Exception):
                        future = asyncio.run_coroutine_threadsafe(
                            session_instance.request_stop(), session_instance.loop
                        )
                        future.result(timeout=1.0)
                    with contextlib.suppress(Exception):
                        future = asyncio.run_coroutine_threadsafe(
                            session_instance._close_ws(timeout=0.5),
                            session_instance.loop,
                        )
                        future.result(timeout=2.0)
                session_thread.join(timeout=1.5)
                if session_thread.is_alive():
                    if not is_active:
                        logger.warning(
                            "Detaching stale inactive session thread and continuing",
                            "🧹",
                        )
                        session_thread = None
                        session_instance = None
                    else:
                        logger.error(
                            "Previous session thread did not finish, aborting new session",
                            "❌",
                        )
                        _session_start_lock.release()
                        return

        audio.ensure_playback_worker_started(config.CHUNK_MS)
        # Clear the playback done event so session waits for wake-up sound
        audio.playback_done_event.clear()
        logger.info("🔧 playback_done_event cleared (waiting for wake-up sound)", "🔧")
        threading.Thread(target=audio.play_random_wake_up_clip, daemon=True).start()
        is_active = True
        interrupt_event = threading.Event()  # Fresh event for each session
        logger.info(f"{source.capitalize()} trigger. Listening...", "🎤")

        def run_session():
            global session_instance, is_active
            try:
                session_instance = BillySession(interrupt_event=interrupt_event)
                session_instance.last_activity[0] = time.time()
                asyncio.run(session_instance.start())
            except Exception as e:
                logger.error(f"Session error: {e}")
            finally:
                move_head("off")
                is_active = False
                session_instance = None  # Clear reference
                logger.info("Waiting for button press...", "🕐")
                # Release lock when session finishes
                with contextlib.suppress(Exception):
                    _session_start_lock.release()  # Lock might already be released

        session_thread = threading.Thread(target=run_session, daemon=True)
        session_thread.start()
        # Lock will be released by the session thread when it finishes
    except Exception as e:
        # If anything goes wrong, release the lock
        logger.error(f"Error starting session: {e}")
        with contextlib.suppress(Exception):
            _session_start_lock.release()


def on_button():
    trigger_session(source="button", require_physical_press=True)


def on_wake_word():
    trigger_session(source="wake word", require_physical_press=False)


def start_loop():
    global wake_word_listener

    audio.detect_devices(debug=config.DEBUG_MODE)
    _ensure_button_hold_thread()

    if config.LOCAL_WAKE_WORD_ENABLED:
        logger.info(
            f"Local wake-word enabled (phrase='{config.LOCAL_WAKE_WORD_PHRASE}')", "🛎️"
        )
        wake_word_listener = WakeWordListener(on_detected=on_wake_word)
        wake_word_listener.start()
    else:
        logger.info("Local wake-word disabled", "🛎️")

    if config.FLAP_ON_BOOT:
        logger.info("Starting Billy startup animation", "🎭")
        move_head("on")
        time.sleep(0.5)
        move_tail(0.3)
        move_tail(0.3)
        move_head("off")
        time.sleep(0.5)
        move_tail(0.3)
        move_tail(0.3)
        logger.info("Billy startup animation complete", "✅")

    button.when_pressed = on_button
    logger.info(
        "Ready. Press button to start a voice session. Press Ctrl+C to quit.", "🎦"
    )
    logger.info("Waiting for button press...", "🕐")
    if config.MOCKFISH:
        logger.info("Mockfish mode: use Enter to simulate button press", "🐟")
        try:
            while True:
                input("Press Enter to simulate button press: ")
                button.is_pressed = True
                on_button()
        except KeyboardInterrupt:
            pass
    else:
        while True:
            _ensure_button_hold_thread()
            time.sleep(0.1)
