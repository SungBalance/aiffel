from PIL import Image
from PIL import ImageFilter

def image_resize(image, height):
  if height == 300:
    return image.resize((800,300))
  else:
    return image.resize((800,600))

def image_rotate(image):
    return image.transpose(Image.ROTATE_180)

def image_change_bw(image):
    return image.convert('L')
