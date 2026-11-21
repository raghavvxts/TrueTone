"""Run the app only for the lifetime of the native Desktop process."""
import os
from pathlib import Path
import runpy
import sys
import threading
import time


def watch_parent(parent_pid):
    # A native force-quit cannot call cleanup, so the child checks its parent too.
    while os.getppid() == parent_pid:
        time.sleep(1)
    os._exit(0)


if __name__ == '__main__':
    parent_pid = int(sys.argv[1])
    if parent_pid <= 1 or os.getppid() != parent_pid:
        raise SystemExit('Start this server through TrueTone.app.')
    threading.Thread(target=watch_parent, args=(parent_pid,), daemon=True).start()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / 'app.py'), run_name='__main__')
