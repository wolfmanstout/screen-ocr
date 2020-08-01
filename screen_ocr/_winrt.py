import asyncio
import threading

import winrt
from winrt.windows.graphics.imaging import BitmapDecoder, BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter

from . import _base


class WinRtReader(_base.BaseReader):
    def __init__(self, radius=100, **kwargs):
        self.radius = radius
        self._engine = OcrEngine.try_create_from_user_profile_languages()

    def read_image(self, image):
        return WinRtScreenContents(self._run_winrt(image))

    def _run_winrt(self, image):
        bytes = image.convert("RGBA").tobytes()
        data_writer = DataWriter()
        data_writer.write_bytes(list(bytes))
        bitmap = SoftwareBitmap(BitmapPixelFormat.RGBA8, image.width, image.height)
        bitmap.copy_from_buffer(data_writer.detach_buffer())
        return self._safely_run_coroutine(self._engine.recognize_async(bitmap))

    @staticmethod
    def _safely_run_coroutine(coroutine):
        """Runs the provided coroutine.

        Works regardless of whether the caller is a coroutine.
        """
        async def wrapper():
            return await coroutine
        result = None
        def run():
            nonlocal result
            result = asyncio.run(wrapper())
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        return result


class WinRtScreenContents(_base.BaseScreenContents):
    def __init__(self, contents):
        self.contents = contents

    def as_string(self):
        return self.contents.text
