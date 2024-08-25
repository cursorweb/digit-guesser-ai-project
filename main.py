import pygame
import numpy as np

import torch
from torchvision import transforms

from neuralnet import NeuralNetwork

import re
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Loaded {DEVICE} device")

model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()


pygame.init()
pygame.font.init()

WID, HEI = 640, 640

screen = pygame.display.set_mode((WID, HEI))
pygame.display.set_caption("Number Recognizer")

font = pygame.font.Font(size=64)

clock = pygame.time.Clock()
running = True

points: list[tuple[int, int]] = []

mouse_pressed = False

save_canvas = pygame.surface.Surface((28, 28))

predicted_text = None


import re


def extract_number(f):
    s = re.findall(r"(?:img)?(\d+)(?:-is\d+)?\.png$", f)
    return int(s[0]) if s else -1


def get_highest():
    list_of_files = os.listdir("./inputs")
    return max(map(extract_number, list_of_files or ["file0.png"]))


index = get_highest()


def rgb2gray(rgb) -> np.ndarray:
    return np.round(
        1 - np.mean(rgb, -1) / 255, decimals=1
    )  # np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def image_to_arr():
    pygame.transform.smoothscale(
        screen.convert_alpha(), (28, 28), dest_surface=save_canvas
    )

    img = pygame.surfarray.pixels3d(save_canvas)
    img = rgb2gray(img)  # convert to 28x28
    img = np.flip(img, axis=-1)
    img = np.rot90(img)

    # mean = np.mean(img)
    # std = np.std(img)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)]
    )

    img = transform(img.copy()).to(torch.float32)  # 1x28x28

    with torch.no_grad():
        img = img.to(DEVICE)
        pred = model(img)

        certainty, guess = pred.exp().max(1)
        certainty, guess = certainty.item(), guess.item()
        return f"{certainty * 100:.2f}% {guess}"


def save_file(num):
    global index
    pygame.transform.smoothscale(
        screen.convert_alpha(), (28, 28), dest_surface=save_canvas
    )

    img = pygame.surfarray.pixels3d(save_canvas)
    img = rgb2gray(img)  # convert to 28x28
    img = np.flip(img, axis=-1)
    img = np.rot90(img)

    index += 1
    pygame.image.save(save_canvas, f"./inputs/img{index}-is{num}.png")


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # LEFT
                mouse_pressed = True
                predicted_text = None
            if event.button == 3:  # RIGHT
                points = []
                predicted_text = None
            if event.button == 2:  # MIDDLE
                predicted_text = image_to_arr()

        if event.type == pygame.KEYDOWN:
            if event.key >= pygame.K_0 and event.key <= pygame.K_9:
                num = event.key - pygame.K_0
                save_file(num)
                points = []

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_pressed = False

    if mouse_pressed:
        pos = pygame.mouse.get_pos()
        points.append(pos)

    screen.fill("white")

    for point in points:
        pygame.draw.circle(screen, (0, 0, 0), point, 25)

    if len(points) == 0:
        predict_font = font.render("Left click to draw", True, (0, 0, 0))
        screen.blit(
            predict_font, (WID / 2 - predict_font.get_width() / 2, HEI / 2 - 50 - 32)
        )
        predict_font = font.render("Middle click to classify", True, (0, 0, 0))
        screen.blit(
            predict_font, (WID / 2 - predict_font.get_width() / 2, HEI / 2 - 32)
        )
        predict_font = font.render("Right click to clear", True, (0, 0, 0))
        screen.blit(
            predict_font, (WID / 2 - predict_font.get_width() / 2, HEI / 2 + 50 - 32)
        )

    if predicted_text:
        predict_font = font.render(predicted_text, True, (15, 15, 255))
        screen.blit(predict_font, (10, 10))

    pygame.display.flip()

    clock.tick(60)

pygame.quit()
