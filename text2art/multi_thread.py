import argparse
import time
import threading

import torch

from AIPainter import AIPainter, device

prompts = "afternoon"
width = 512  # @param {type:"number"}
height = 512  # @param {type:"number"}
model_selected = "vqgan_imagenet_f16_16384"  # @param ["vqgan_imagenet_f16_16384", "vqgan_imagenet_f16_1024", "wikiart_16384", "coco", "faceshq", "sflckr"]
display_frequency = 50  # @param {type:"number"}
initial_image = ""  # @param {type:"string"}
target_images = ""  # @param {type:"string"}
seed = -1  # @param {type:"number"}
max_iterations = 100  # @param {type:"number"}
input_images = ""

model_names = {"vqgan_imagenet_f16_16384": 'ImageNet 16384', "vqgan_imagenet_f16_1024": "ImageNet 1024",
               "wikiart_16384": "WikiArt 16384", "coco": "COCO-Stuff", "faceshq": "FacesHQ",
               "sflckr": "S-FLCKR"}
model_name = model_names[model_selected]

if seed == -1:
    seed = None
if initial_image == "None":
    initial_image = None
if target_images == "None" or not target_images:
    target_images = []
else:
    target_images = target_images.split("|")
    target_images = [image.strip() for image in target_images]

if initial_image or target_images != []:
    input_images = True

prompts = [frase.strip() for frase in prompts.split("|")]
if prompts == ['']:
    prompts = []

args = argparse.Namespace(
    prompts=prompts,
    image_prompts=target_images,
    noise_prompt_seeds=[],
    noise_prompt_weights=[],
    size=[width, height],
    init_image=initial_image,
    init_weight=0.,
    clip_model='ViT-B/32',
    vqgan_config=f'{model_selected}.yaml',
    vqgan_checkpoint=f'{model_selected}.ckpt',
    step_size=0.1,
    cutn=64,
    cut_pow=1.,
    display_freq=display_frequency,
    seed=seed,
    max_iterations=max_iterations,
)

print('Using device:', device)
if prompts:
    print('Using text prompt:', prompts)
if target_images:
    print('Using image prompts:', target_images)
if args.seed is None:
    seed = torch.seed()
else:
    seed = args.seed
torch.manual_seed(seed)
print('Using seed:', seed)

painter = AIPainter(args)
painter_thread = threading.Thread(target=painter.run, name="painterThread")

painter_thread.start()

print("Run!")

while painter_thread.is_alive():
    print("\n\t" + str(painter.getCurIter()) + "\n")
    time.sleep(4)

# with open('./temp.txt', 'w') as f:
#     while painter_thread.is_alive():
#         f.write("\n\t" + cur_iter + "\n")
#         time.sleep(4)
#
print("What fuck!")
