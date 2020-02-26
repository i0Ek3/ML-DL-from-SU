#!/usr/bin/env python3

# 缩放pic到50%,并模糊图片

from PIL import Image, ImageFilter

im = Image.open('man.jpg')
w, h = im.size
print('Original image size: %sx%s' % (w, h))
im.thumbnail((w//2, h//2))
print('Resize image to: %sx%s' % (w//2, h//2))
i3 = im.filter(ImageFilter.BLUR)
i3.save('blur.jpg', 'jpeg')
