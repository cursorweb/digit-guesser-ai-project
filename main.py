import pygame
import numpy as np

import torch

from neuralnet import NeuralNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Loaded {DEVICE} device")

model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()


pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((640, 640))
pygame.display.set_caption("Number Recognizer")

font = pygame.font.Font(size=64)

clock = pygame.time.Clock()
running = True

points: list[tuple[int, int]] = []

mouse_pressed = False

save_canvas = pygame.surface.Surface((28, 28))

predicted_text = None


def rgb2gray(rgb) -> np.ndarray:
    return np.round(
        1 - np.mean(rgb, -1) / 255, decimals=1
    )  # np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def image_to_arr():
    pygame.transform.smoothscale(
        screen.convert_alpha(), (28, 28), dest_surface=save_canvas
    )

    pygame.image.save(save_canvas, "number2.png")

    img = pygame.surfarray.pixels3d(save_canvas)
    img = rgb2gray(img)  # convert to 28x28
    img = np.rot90(img)
    img = np.expand_dims(img, axis=0)  # 1x28x28

    img = torch.as_tensor(img.copy()).to(torch.float32)
    with torch.no_grad():
        img = img.to(DEVICE)
        pred = model(img)

        certainty, guess = pred.exp().max(1)
        certainty, guess = certainty.item(), guess.item()
        return f"{certainty * 100:.2f}% {guess}"
        # print("guess:", guess, "certainty:", certainty)


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

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_pressed = False

    if mouse_pressed:
        pos = pygame.mouse.get_pos()
        points.append(pos)

    screen.fill("white")

    for point in points:
        pygame.draw.circle(screen, (0, 0, 0), point, 30)

    if predicted_text:
        predict_font = font.render(predicted_text, True, (15, 15, 255))
        screen.blit(predict_font, (10, 10))

    pygame.display.flip()

    clock.tick(60)

pygame.quit()
