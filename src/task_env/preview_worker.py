from __future__ import annotations

import queue
import threading
from typing import Any

import rerun as rr

from robot.utils.base.data_handler import debug_print


PreviewItem = tuple[int, dict[str, bytes]]


class PreviewWorker:
    def __init__(
        self,
        recording: rr.RecordingStream,
    ) -> None:
        self.recording = recording

        self._queue: queue.Queue[PreviewItem] = queue.Queue(
            maxsize=1
        )
        self._stop_event = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            name="rerun-preview-worker",
            daemon=True,
        )

        self.submitted_frames = 0
        self.logged_frames = 0
        self.dropped_frames = 0
        self.error_count = 0

    def start(self) -> None:
        self._thread.start()

        debug_print(
            "PREVIEW",
            (
                "PreviewWorker started: "
                f"thread_alive={self._thread.is_alive()}, "
                "queue_size=1"
            ),
            "INFO",
        )

    def submit(
        self,
        frame_index: int,
        sensor_data: dict[str, Any],
    ) -> None:
        if self._stop_event.is_set():
            return

        cameras: dict[str, bytes] = {}

        if self.submitted_frames == 0:
            debug_print(
                "PREVIEW",
                (
                    f"First submit: frame={frame_index}, "
                    f"sensor_keys={list(sensor_data.keys())}"
                ),
                "INFO",
            )

        for cam_name, cam_data in sensor_data.items():
            if not isinstance(cam_data, dict):
                debug_print(
                    "PREVIEW",
                    (
                        f"{cam_name}: expected dict, "
                        f"got {type(cam_data).__name__}"
                    ),
                    "WARNING",
                )
                continue

            image = cam_data.get("color")

            if image is None:
                debug_print(
                    "PREVIEW",
                    f"{cam_name}: color is None",
                    "WARNING",
                )
                continue

            if isinstance(image, bytes):
                jpeg = image
            elif isinstance(image, bytearray):
                jpeg = bytes(image)
            elif isinstance(image, memoryview):
                jpeg = image.tobytes()
            else:
                debug_print(
                    "PREVIEW",
                    (
                        f"{cam_name}: unsupported color type "
                        f"{type(image).__name__}"
                    ),
                    "WARNING",
                )
                continue

            # JPEG 文件应以 FF D8 开头。
            if len(jpeg) < 2 or jpeg[:2] != b"\xff\xd8":
                debug_print(
                    "PREVIEW",
                    (
                        f"{cam_name}: data is not JPEG, "
                        f"type={type(image).__name__}, "
                        f"size={len(jpeg)}, "
                        f"header={jpeg[:8].hex()}"
                    ),
                    "WARNING",
                )
                continue

            cameras[cam_name] = jpeg

        if not cameras:
            debug_print(
                "PREVIEW",
                (
                    f"No valid JPEG images found at "
                    f"frame={frame_index}"
                ),
                "WARNING",
            )
            return

        if self.submitted_frames == 0:
            camera_sizes = {
                name: len(data)
                for name, data in cameras.items()
            }

            debug_print(
                "PREVIEW",
                (
                    f"First valid preview frame: "
                    f"cameras={camera_sizes}"
                ),
                "INFO",
            )

        item: PreviewItem = (
            int(frame_index),
            cameras,
        )

        try:
            self._queue.put_nowait(item)
            self.submitted_frames += 1
            return
        except queue.Full:
            pass

        # 丢弃旧预览帧。
        try:
            self._queue.get_nowait()
            self.dropped_frames += 1
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(item)
            self.submitted_frames += 1
        except queue.Full:
            self.dropped_frames += 1

    def _run(self) -> None:
        debug_print(
            "PREVIEW",
            "PreviewWorker run loop entered.",
            "INFO",
        )

        while not self._stop_event.is_set():
            try:
                frame_index, cameras = self._queue.get(
                    timeout=0.2
                )
            except queue.Empty:
                continue

            try:
                if self.logged_frames == 0:
                    debug_print(
                        "PREVIEW",
                        (
                            f"Worker received first frame: "
                            f"frame={frame_index}, "
                            f"cameras={list(cameras.keys())}"
                        ),
                        "INFO",
                    )

                # 使用显式 RecordingStream。
                # set_time 必须和 log 位于同一线程。
                self.recording.set_time(
                    "frame",
                    sequence=frame_index,
                )

                for cam_name, jpeg in cameras.items():
                    self.recording.log(
                        f"cameras/{cam_name}",
                        rr.EncodedImage(
                            contents=jpeg,
                            media_type="image/jpeg",
                        ),
                    )

                self.logged_frames += 1

                if (
                    self.logged_frames == 1
                    or self.logged_frames % 100 == 0
                ):
                    debug_print(
                        "PREVIEW",
                        (
                            f"Logged preview frame={frame_index}, "
                            f"logged={self.logged_frames}, "
                            f"dropped={self.dropped_frames}"
                        ),
                        "INFO",
                    )

            except Exception as exc:
                self.error_count += 1

                debug_print(
                    "PREVIEW",
                    (
                        f"Rerun log failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "ERROR",
                )

    def close(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout)

        debug_print(
            "PREVIEW",
            (
                f"PreviewWorker stopped: "
                f"submitted={self.submitted_frames}, "
                f"logged={self.logged_frames}, "
                f"dropped={self.dropped_frames}, "
                f"errors={self.error_count}, "
                f"thread_alive={self._thread.is_alive()}"
            ),
            "INFO",
        )