import pygame
from math import *
from random import randint

pygame.init()

name = input('Picture name (default perlin.bmp) -> ')
if name == '':
    name = 'Perlin.bmp'

img = pygame.image.load(name)

num = input('Erosion strength (default 4) -> ')
try:
    num = int(num)
except:
    num = 4

arr = [[img.get_at([i, j])[0] for j in range(img.get_height())]
       for i in range(img.get_width())]

log = open('log.txt', 'w')

def erode(x, y, steps=50):
    if not (0 <= x < len(arr) and 0 <= y < len(arr[x])):
        return
    poss = []
    for dirr in [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]:
        X, Y = x + dirr[0], y + dirr[1]
        if (0 <= X < len(arr) and 0 <= Y < len(arr[X])):
            if arr[x][y] > arr[X][Y]:
                k = 0.99
                arr[x][y] = arr[X][Y] + (arr[x][y] - arr[X][Y]) * k
                #arr[X][Y] = arr[X][Y] + (arr[x][y] - arr[X][Y]) * 0.01
                poss.append([X, Y])
    if steps > 0 and len(poss) > 0:
        pos = poss[randint(0, len(poss) - 1)]
        erode(pos[0], pos[1], steps - 1)

def shuffle(arr):
    answ = []
    ar = arr.copy()
    for i in range(len(arr)):
        answ.append(ar.pop(randint(0, len(ar) - 1)))
    return answ

scr = pygame.display.set_mode([len(arr), len(arr[0])])

positions = [[x // len(arr), x % len(arr)] for x in range(len(arr) * len(arr[0]))]
print(len(positions))

for _ in range(num):
    shuf = shuffle(positions)[:]
    for i in range(len(shuf)):
        cX, cY = shuf[i]
        erode(cX, cY)
        if i % 1000 == 0:
            #print(i)
            for i in range(len(arr)):
                for j in range(len(arr[0])):
                    img.set_at([i, j], [min(255, round(arr[i][j]))] * 3)
            scr.blit(img, [0, 0])
            pygame.display.update()

pygame.image.save(scr, 'Result.bmp')

log.close()
