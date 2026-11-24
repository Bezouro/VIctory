"""Replay a RenderDoc frame, export the presented Vulkan image, and time warm replays.

Run inside RenderDoc, which supplies the renderdoc Python module:
    VICTORY_REPLAY_FILE=scene.rdc VICTORY_REPLAY_IMAGE=scene.png \
        qrenderdoc --python scripts/replay-frame.py
Set VICTORY_REPLAY_LOOPS to the number of timed repetitions (default 5).
"""

import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import renderdoc as rd


def presented_image(controller):
    stack = list(controller.GetRootActions())
    actions = []
    while stack:
        action = stack.pop()
        actions.append(action)
        stack.extend(action.children)
    present = max(
        (a for a in actions if a.flags & rd.ActionFlags.Present),
        key=lambda a: a.eventId,
        default=None,
    )
    if present is None:
        raise ValueError("capture contains no presentation")
    textures = {texture.resourceId: texture for texture in controller.GetTextures()}
    for candidate in (present.copyDestination, present.copySource):
        if candidate in textures:
            return actions, present, textures[candidate]
    swap = [t for t in textures.values() if t.creationFlags & rd.TextureCategory.SwapBuffer]
    if len(swap) != 1:
        raise ValueError("cannot identify the presented swapchain image")
    return actions, present, swap[0]


def replay(path, output, count):
    if count < 1:
        raise ValueError("VICTORY_REPLAY_LOOPS must be positive")
    if output.suffix.lower() != ".png" or output.resolve() == path.resolve():
        raise ValueError("choose a separate .png file for the replayed image")
    capture = rd.OpenCaptureFile()
    controller = None
    started = time.perf_counter()
    try:
        result = capture.OpenFile(str(path), "", None)
        if result != rd.ResultCode.Succeeded:
            raise RuntimeError(str(result))
        result, controller = capture.OpenCapture(rd.ReplayOptions(), None)
        if result != rd.ResultCode.Succeeded:
            raise RuntimeError(str(result))
        opened = time.perf_counter() - started
        actions, present, texture = presented_image(controller)

        def replay_once():
            start = time.perf_counter()
            controller.SetFrameEvent(present.eventId, True)
            # A pixel read waits for the GPU. Report wall time including that read.
            controller.PickPixel(texture.resourceId, texture.width // 2, texture.height // 2,
                                 rd.Subresource(), rd.CompType.Typeless)
            return (time.perf_counter() - start) * 1000

        replay_once()
        timings = [replay_once() for _ in range(count)]
        save = rd.TextureSave()
        save.resourceId = texture.resourceId
        save.destType = rd.FileType.PNG
        save.alpha = rd.AlphaMapping.Discard
        result = controller.SaveTexture(save, str(output))
        if result != rd.ResultCode.Succeeded:
            raise RuntimeError(str(result))
        return {
            "size": [texture.width, texture.height],
            "event": present.eventId,
            "draws": sum(bool(a.flags & rd.ActionFlags.Drawcall) for a in actions),
            "dispatches": sum(bool(a.flags & rd.ActionFlags.Dispatch) for a in actions),
            "open_seconds": opened,
            "replay_and_pixel_read_ms": timings,
            "median_ms": statistics.median(timings),
        }
    finally:
        if controller:
            controller.Shutdown()
        capture.Shutdown()


try:
    output = Path(os.environ["VICTORY_REPLAY_IMAGE"])
    report = replay(Path(os.environ["VICTORY_REPLAY_FILE"]), output,
                    int(os.environ.get("VICTORY_REPLAY_LOOPS", "5")))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
except Exception:
    traceback.print_exc(file=sys.__stderr__)
    sys.__stderr__.flush()
    os._exit(1)  # qrenderdoc consumes SystemExit as a successful script exit.
sys.exit(0)
