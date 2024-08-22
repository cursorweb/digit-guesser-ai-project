import pygame

pygame.init()
screen = pygame.display.set_mode((640, 640))
pygame.display.set_caption("Number Recognizer")

clock = pygame.time.Clock()
running = True

points = []

mouse_pressed = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # LEFT
                mouse_pressed = True
            if event.button == 3:  # RIGHT
                points = []

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_pressed = False

    if mouse_pressed:
        pos = pygame.mouse.get_pos()
        points.append(pos)

    screen.fill("white")

    for point in points:
        pygame.draw.circle(screen, (0, 0, 0), point, 20)

    pygame.display.flip()

    clock.tick(60)

pygame.quit()
