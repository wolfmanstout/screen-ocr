"""Base classes."""


class BaseReader(object):
    """Reads on-screen text using OCR."""

    def read_nearby(self, screen_coordinates):
        """Return ScreenContents nearby the provided coordinates."""
        raise NotImplementedError()

    def read_image(self, image):
        """Return ScreenContents of the provided image."""
        raise NotImplementedError()


class BaseScreenContents(object):
    """OCR'd contents of a portion of the screen."""

    def as_string(self):
        """Return the contents formatted as a string."""
        raise NotImplementedError()

    def find_nearest_word_coordinates(self, word, cursor_position):
        """Return the coordinates of the nearest instance of the provided word.

        Uses fuzzy matching.

        Arguments:
        word: The word to search for.
        cursor_position: "before", "middle", or "after" (relative to the matching word)
        """
        raise NotImplementedError()
