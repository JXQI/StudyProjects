import numpy as np
import torch
from PIL import Image,ImageDraw

# image=Image.open("/Users/jinxiaoqiang/Desktop/1.png")
# img=image.convert("RGBA")
# tmp = Image.new('RGBA', img.size, (0,0,0,0))
# draw = ImageDraw.Draw(tmp)
# # Determine the bounding box of the largest possible semi-transparent
# # square rectangle centered on the temporary image and draw it.
# if img.size[0] > img.size[1]:
#     size = img.size[1]
#     llx, lly = (img.size[0] - img.size[1]) // 2, 0
# else:
#     size = img.size[0]
#     llx, lly = 0, (img.size[1] - img.size[0]) // 2
#
# # Add one to upper point because second point is just outside the drawn
# # rectangle.
# urx, ury = llx + size + 1, lly + size + 1
# urx,ury= 100,200
# draw.rectangle(((llx, lly), (urx, ury)), fill=(0,0,0,127))
#
# # Alpha composite the two images together.
# img = Image.alpha_composite(img, tmp)
# img = img.convert("RGB") # Remove alpha for saving in jpg format.
# # img.show()
# img=img.convert("L")
# img.show()
# # image = Image.fromarray(array, mode='RGB') #打开图片
# img.save("/Users/jinxiaoqiang/Desktop/2.png")

image=Image.open("/Users/jinxiaoqiang/Desktop/1.png")
image=image.convert("L")
print(image.size)
img=Image.new("L",image.size)
img.show()