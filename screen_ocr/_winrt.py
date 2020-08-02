import asyncio
import threading
from concurrent import futures

from . import _base
from . import _utils


class WinRtReader(_base.BaseReader):
    def __init__(self, radius=100, **kwargs):
        self.radius = radius
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._executor.submit(self._init_winrt)

    def read_nearby(self, screen_coordinates):
        screenshot, bounding_box = _utils.screenshot_nearby(screen_coordinates, self.radius)
        result = self._executor.submit(lambda: self._run_ocr_sync(screenshot)).result()
        return WinRtScreenContents(result, (bounding_box[0], bounding_box[1]), screenshot, self._executor)
        
    def read_image(self, image):
        result = self._executor.submit(lambda: self._run_ocr_sync(image)).result()
        return WinRtScreenContents(result, (0, 0), image, self._executor)

    def _init_winrt(self):
        import winrt
        import winrt.windows.graphics.imaging as imaging
        import winrt.windows.media.ocr as ocr
        import winrt.windows.storage.streams as streams
        engine = ocr.OcrEngine.try_create_from_user_profile_languages()
        async def run_ocr_async(image):
            bytes = image.convert("RGBA").tobytes()
            data_writer = streams.DataWriter()
            data_writer.write_bytes(list(bytes))
            bitmap = imaging.SoftwareBitmap(imaging.BitmapPixelFormat.RGBA8, image.width, image.height)
            bitmap.copy_from_buffer(data_writer.detach_buffer())
            return await engine.recognize_async(bitmap)
        self._run_ocr_async = run_ocr_async

    def _run_ocr_sync(self, image):
        return asyncio.run(self._run_ocr_async(image))


class WinRtScreenContents(_base.BaseScreenContents):
    def __init__(self, contents, offset, screenshot, executor):
        self._contents = contents
        self._offset = offset
        self.screenshot = screenshot
        self._executor = executor

    def as_string(self):
        return self._executor.submit(lambda: self._contents.text).result()

    def find_nearest_word_coordinates(self, word, cursor_position):
        if cursor_position not in ("before", "middle", "after"):
            raise ValueError("cursor_position must be either before, middle, or after")
        def run():
            for line in self._contents.lines:
                for result_word in line.words:
                    if word in result_word.text:
                        box = result_word.bounding_rect
                        return (int(box.x + box.width / 2.0 + self._offset[0]),
                                int(box.y + box.height / 2.0 + self._offset[1]))
        return self._executor.submit(run).result()
