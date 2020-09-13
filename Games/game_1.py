import pygame

pygame.init()

win = pygame.display.set_mode((500, 480))
pygame.display.set_caption('First Game')

bg = pygame.image.load('bg.jpg')
image_list_l = ['L1.png', 'L2.png', 'L3.png', 'L4.png', 'L5.png', 'L6.png', 'L7.png', 'L8.png', 'L9.png']
walk_left = []

[walk_left.append(pygame.image.load(image_left)) for image_left in image_list_l]

image_list_r = ['R1.png', 'R2.png', 'R3.png', 'R4.png', 'R6.png', 'R7.png', 'R8.png', 'R9.png']
walk_right = []

[walk_right.append(pygame.image.load(image_right)) for image_right in image_list_r]

char = pygame.image.load('standing.png')

clock = pygame.time.Clock()
x = 50
y = 400
width = 64
height = 64
vel = 5
isJump = False
jump_count = 10
left = False
right = False

def redraw_window():
    global walk_count
    win.blit(bg, (0, 0))
    if walk_count + 1 >= 27:
        walk_count = 0
    if left:
        win.blit(walk_left[walk_count // 3], (x, y))
        walk_count += 1
    elif right:
        win.blit(walk_right[walk_count // 3], (x, y))
    else:
        win.blit(char, (x, y))
    pygame.display.update()

run = True
while run:
    clock.tick(27)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and x > vel:
        x -= vel
        left = True
        right = False
    elif keys[pygame.K_RIGHT] and x < 500 - width - vel:
        x += vel
        right = True
        left = False
    else:
        right = False
        left =False
        walk_count = 0
    if not(isJump):
       if keys[pygame.K_SPACE]:
           left = False
           right = False
           isJump = True
           walk_count = 0
    else:
        if jump_count >= -10:
            neg = 1
            if jump_count < 0:
                neg = -1
            y -= (jump_count ** 2) * 0.5 * neg
            jump_count -= 1
        else:
            isJump = False
            jump_count = 10
    redraw_window()

pygame.quit()
