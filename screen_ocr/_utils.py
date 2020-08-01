from PIL import ImageGrab


def screenshot_nearby(screen_coordinates, radius):
    # TODO Consider cropping within grab() for performance. Requires knowledge
    # of screen bounds.
    screenshot = ImageGrab.grab()
    bounding_box = (max(0, screen_coordinates[0] - radius),
                    max(0, screen_coordinates[1] - radius),
                    min(screenshot.width, screen_coordinates[0] + radius),
                    min(screenshot.height, screen_coordinates[1] + radius))
    screenshot = screenshot.crop(bounding_box)
    return screenshot, bounding_box


def distance_squared(x1, y1, x2, y2):
    x_dist = (x1 - x2)
    y_dist = (y1 - y2)
    return x_dist * x_dist + y_dist * y_dist
