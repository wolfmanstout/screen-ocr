# import gc
import asyncio
import threading
from concurrent import futures

from . import _base


class WinRtBackend(_base.OcrBackend):
    def __init__(self):
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._executor.submit(self._init_winrt).result()

    def run_ocr(self, image):
        return self._executor.submit(lambda: self._run_ocr_sync(image)).result()

    def _init_winrt(self):
        import winrt
        import winrt.windows.graphics.imaging as imaging
        import winrt.windows.media.ocr as ocr
        import winrt.windows.storage.streams as streams
        engine = ocr.OcrEngine.try_create_from_user_profile_languages()
        async def run_ocr_async(image):
            bytes = image.convert("RGBA").tobytes()
            data_writer = streams.DataWriter()
            bytes_list = list(bytes)
            del bytes
            # Needed when testing on large files on 32-bit.
            # gc.collect()
            data_writer.write_bytes(bytes_list)
            del bytes_list
            bitmap = imaging.SoftwareBitmap(imaging.BitmapPixelFormat.RGBA8, image.width, image.height)
            bitmap.copy_from_buffer(data_writer.detach_buffer())
            del data_writer
            result = await engine.recognize_async(bitmap)
            lines = [_base.OcrLine([_base.OcrWord(word.text,
                                                  word.bounding_rect.x,
                                                  word.bounding_rect.y,
                                                  word.bounding_rect.width,
                                                  word.bounding_rect.height)
                                    for word in line.words])
                     for line in result.lines]
            return _base.OcrResult(lines)
        self._run_ocr_async = run_ocr_async

    def _run_ocr_sync(self, image):
        return asyncio.run(self._run_ocr_async(image))
