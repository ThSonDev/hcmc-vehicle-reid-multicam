"""Headless-aware wrappers around OpenCV's GUI calls.

Set REID_HEADLESS=1 (run.py --headless passes it down) to disable all windows, so the
pipeline runs over SSH / on a box with no display. Detection, tracking, matching,
res-file writing and JSONL logging are unaffected — only the live preview is skipped.

Drop-in for the cv2 GUI calls the pipeline uses: same names, no-ops when headless.
"""
import os

import cv2

HEADLESS = os.environ.get("REID_HEADLESS") == "1"


def imshow(window, frame):
    if not HEADLESS:
        cv2.imshow(window, frame)


def waitKey(delay=1):
    """Pump the GUI event loop and return the pressed key, or -1 when headless
    (so `waitKey(1) == ord('q')` is simply never true and the loop keeps running)."""
    if HEADLESS:
        return -1
    return cv2.waitKey(delay)


def namedWindow(window):
    if not HEADLESS:
        cv2.namedWindow(window)


def setMouseCallback(window, callback):
    if not HEADLESS:
        cv2.setMouseCallback(window, callback)


def destroyAllWindows():
    if not HEADLESS:
        cv2.destroyAllWindows()
