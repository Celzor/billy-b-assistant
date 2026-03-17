import shutil
import signal
import sys
import threading
import traceback
from pathlib import Path

from core.logger import logger


# --- Ensure .env exists ---
def ensure_env_file():
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists():
        if env_example_path.exists():
            shutil.copy(env_example_path, env_path)
            print("✅ .env file created from .env.example")
            print(
                "⚠️  Please review the .env file and update your API key and other settings."
            )
        else:
            print("❌ Neither .env nor .env.example found. Exiting.")
            sys.exit(1)


ensure_env_file()

# --- Now load env ---
from dotenv import load_dotenv


load_dotenv()

# --- Imports that might use environment variables ---
from pathlib import Path

import core.button
from core.audio import playback_queue

# --- Reload logger level after environment is loaded ---
from core.logger import reload_log_level
from core.movements import start_motor_watchdog
from core.mqtt import start_mqtt, stop_mqtt


current_level = reload_log_level()
print(f"🔧 Log level set to: {current_level.name}")


def signal_handler(sig, frame):
    logger.info("Exiting cleanly (signal received).", "👋")
    playback_queue.put(None)
    from core.movements import cleanup_gpio

    if getattr(core.button, "wake_word_listener", None):
        core.button.wake_word_listener.stop()

    cleanup_gpio()
    stop_mqtt()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load default user profile BEFORE starting button loop
    # This ensures the persona manager is set to the correct persona before any sessions start
    from core.profile_manager import user_manager

    user_manager.load_default_user()

    threading.Thread(target=start_mqtt, daemon=True).start()
    start_motor_watchdog()
    core.button.start_loop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Unhandled exception occurred:", e)
        traceback.print_exc()
        from core.movements import cleanup_gpio

        if getattr(core.button, "wake_word_listener", None):
            core.button.wake_word_listener.stop()

        cleanup_gpio()
        stop_mqtt()
        sys.exit(1)
