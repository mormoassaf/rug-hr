from PIL import Image, ImageFont, ImageDraw


def load_font(font_path, font_size):
    return ImageFont.truetype(font_path, font_size)