from PIL import Image
import cv2
import os
import tinify
from tqdm import tqdm

tinify.key = "pdpNtbT0t6sJFMnBQh5dNl4ChXlGmFvW"

# 打开图像
# img = Image.open('/Users/Maple/Downloads/original.jpg')
# w, h = img.size
# resize_w, resize_h = 512, 288
# # 方法一：使用resize改变图片分辨率，但是图片内容并不丢失，不是裁剪
# # img_resize = img.resize((int(w * compress_rate), int(h * compress_rate)))
# img_resize = img.resize((resize_w, resize_h))
# img_resize.save('/Users/Maple/Downloads/original-compress.jpg')


fileName = '实验结果对比-图像增强'
source = tinify.from_file(f"/Users/Maple/Downloads/{fileName}.png")
source.to_file(f"/Users/Maple/Downloads/{fileName}-compress.png")

# directory_name = '多文本生成图像'
# src_directory = '/Users/Maple/Downloads/毕设对比/' + directory_name
# dest_directory = '/Users/Maple/Downloads/毕设对比-compress/' + directory_name
# src_imgs = os.listdir(src_directory)
# src_imgs = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), src_imgs))
#
# for img in tqdm(src_imgs):
#     source = tinify.from_file(os.path.join(src_directory, img))
#     source.to_file(os.path.join(dest_directory, img))
