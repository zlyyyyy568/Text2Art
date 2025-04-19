# import torch
# from CLIP import clip
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
#
#
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model, preprocess = clip.load('ViT-B/32', device=device)
#
# original_image = Image.open('./samples/Cartoon.png')
# image = preprocess(original_image).unsqueeze(0).to(device)
# # text = clip.tokenize(['a person', 'a woman', 'a man', 'a cartoon picture']).to(device)
# text = clip.tokenize(['a woman', 'a dog', 'a pet']).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     test_features = model.encode_text(text)
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("label probs: ", probs)

class Demo:
    pass

def f():
    demo = Demo()
    demo.info = '我是Demo!'

f()
print(demo.info)
