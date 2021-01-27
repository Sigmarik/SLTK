import pygame
from math import *
import time
import threading

INF = 10 ** 2
PI = 3.1415926535

SKYCOL = [100, 100, 250]
DRAWDIST = 1000

def MOD(x, m):
    if x == 0:
        return 0
    else:
        return (abs(x) % m) * int(abs(x) / x)

class vertex:
    pos = [0, 0, 0]
    def __init__(self, X = 0, Y = 0, Z = 0):
        if type(X) in [type([1, 2]), type((3, 4))]:
            self.pos = list(X).copy()
        else:
            self.pos = [X, Y, Z]
    def X(self):
        return self.pos[0]
    def Y(self):
        return self.pos[1]
    def Z(self):
        return self.pos[2]
    def len(self):
        return sqrt(sum([self.pos[i] * self.pos[i] for i in range(3)]))
    def __eq__(A, B):
        return list(A.pos) == list(B.pos)
    def __ne__(A, B):
        return list(A.pos) != list(B.pos)
    def __add__(A, B):
        return vertex([A.pos[i] + B.pos[i] for i in range(3)])
    def __sub__(A, B):
        return vertex([A.pos[i] - B.pos[i] for i in range(3)])
    def __mul__(A, B):
        return vertex([A.pos[i] * B for i in range(3)])
    def __truediv__(A, B):
        return vertex([A.pos[i] / B for i in range(3)])
    def __iadd__(self, B):
        self = self + B
    def __isub__(self, B):
        self = self - B
    def __imul__(self, B):
        self = self * B
    def __itruediv__(self, B):
        self = self / B
    def __neg__(self):
        return self * -1
    def get_y(self):
        return atan2(self.Z(), self.X())
    def get_z(self):
        return atan2(self.Y(), self.X())
    def get_rotation(self):
        Y = atan2(self.Z(), self.X())
        X = 0
        Z = atan2(self.Y(), self.X())
        return [X, Y, Z]
    def apply_rot_x(self, deg):
        X = self.X()
        D = vertex(self.Y(), self.Z()).len()
        Z = cos(deg) * D
        Y = sin(deg) * D
        return vertex(X, Y, Z)
    def apply_rot_y(self, deg):
        Y = self.Y()
        D = vertex(self.X(), self.Z()).len()
        X = cos(deg) * D
        Z = sin(deg) * D
        return vertex(X, Y, Z)
    def apply_rot_z(self, deg):
        Z = self.Z()
        D = vertex(self.X(), self.Y()).len()
        X = cos(deg) * D
        Y = sin(deg) * D
        return vertex(X, Y, Z)
    def add_rot_y(self, deg):
        return self.apply_rot_y(self.get_y() + deg)
    def add_rot_z(self, deg):
        return self.apply_rot_z(self.get_z() + deg)
    def apply_rotation_old(self, rot):
        D = self.len()
        Y = sin(rot[2]) * cos(rot[1]) * D
        X = cos(rot[2]) * cos(rot[1]) * D
        Z = sin(rot[1]) * D
        return vertex(X, Y, Z)
    def apply_rotation(self, rot):
        D = self.len()
        root = vertex(D).apply_rot_y(rot[1]).apply_rot_z(rot[2])
        return root
    def set_rotation(self, rot):
        self = self.apply_rotation(rot)
    def rotated(self, rot):
        result = self.add_rot_y(rot[1]).add_rot_z(rot[2])
        return result
    def rotate(self, rot):
        self = self.rotated(rot)
    def rotated_origin(self, orig, rot):
        return orig + (self - orig).rotated(rot)
    def rotate_origin(self, orig, rot):
        self = self.rotated_origin(orig, rot)
    def apply_rotation_origin(self, orig, rot):
        D = (self - orig).apply_rotation(rot)
        return orig + D
    def set_rotation_origin(self, orig, rot):
        self = self.apply_rotation_origin(rot, orig)

def dist(A, B = vertex(0, 0, 0)):
    if B != vertex(0, 0, 0):
        return dist(A - B)
    else:
        return sqrt(sum([A.pos[i] * A.pos[i] for i in range(3)]))

def scalmul(A, B):
    return A.X() * B.X() + A.Y() * B.Y()

def vecmul(A, B):
    return A.X() * B.Y() - A.Y() * B.X()

class camera:
    pos = vertex()
    rot = [0] * 3
    scr_dist = 600
    resolution = [600, 600]
    def __init__(self, pos = vertex(), rot = [0] * 3, sdist = 600, resol = [600, 600]):
        self.pos = pos
        self.rot = rot.copy()
        self.scr_dist = sdist
        self.resolution = resol.copy()
    def forward(self):
        return sin(self.rot[0])

def drop(vert, cam):
    #vis_vert = (vert - cam.pos).rotated([-x for x in cam.rot])
    vis_vert = (vert - cam.pos).add_rot_z(-cam.rot[2]).add_rot_y(-cam.rot[1])
    if vis_vert.X() > 0:
        ky = vis_vert.Y() / vis_vert.X()
        kz = vis_vert.Z() / vis_vert.X()
        return [ky * cam.scr_dist, -kz * cam.scr_dist]
    else:
        answ = vertex(vis_vert.Y() * INF, -vis_vert.Z() * INF, -1)
        answ = answ * (cam.resolution[0] + cam.resolution[1]) / answ.len()
        return answ.pos
    

class polygon:
    verts = [vertex(0, 0)] * 3
    color = [0, 100, 0]
    dirrection = [0, 0, 0]
    def __init__(self, vers, col = [0, 100, 0]):
        self.verts = vers.copy()
        self.color = col.copy()
    def rotate(self, rot, orig = vertex()):#ERR
        for i in range(len(self.verts)):
            self.verts[i] = self.verts[i].rotated_origin(orig, rot)#<--
        self.dirrection = [self.dirrection[i] + rot[i] for i in range(3)]
    def set_rot(self, rot, orig = vertex()):
        drot = [rot[i] - self.dirrection[i] for i in range(3)]
        self.rotate(drot, orig)
        self.dirrection = rot.copy()
        return self
    def draw(self, cam, picture):
        answ = []
        for vert in self.verts:
            pos = drop(vert, cam)
            answ.append(pos)
        A = vertex(*answ[0])
        B = vertex(*answ[1])
        C = vertex(*answ[2])
        if vecmul(B - A, C - A) < 0 and any((x.pos[2] >= 0 and x.pos[0] + x.pos[1] < 10000) for x in (A, B, C)):
            pygame.draw.polygon(picture, mix(SKYCOL, self.color, 0 * min(self.dist(cam) / DRAWDIST, 1)),
                                [(MOD(int(p[0] + cam.resolution[0] // 2), 10000), MOD(int(p[1] + cam.resolution[1] // 2), 10000)) for p in answ])
    def dist(self, cam):
        return sum([dist(x, cam.pos) for x in self.verts]) / 3
    def apply_to_all(self, func, args):
        for i in range(len(self.verts)):
            self.verts[i] = func(self.verts[i], *args)
        return self
    def __lt__(A, B):
        return True
        
class mesh:
    polygons = []
    def __init__(self, polys):
        self.polygons = polys.copy()
    def draw(self, cam, picture):
        answ = []
        for D, pol in sorted([[p.dist(cam), p] for p in self.polygons])[::-1]:
            pol.draw(cam, picture)
    def rotate(self, rot, orig = vertex()):
        for i in range(len(self.polygons)):
            self.polygons[i].rotate(rot, orig)
    def set_rot(self, rot, orig = vertex()):
        for i in range(len(self.polygons)):
            self.polygons[i] = self.polygons[i].set_rot(rot, orig)
        #print(self.polygons)
    def copy(self):
        return mesh(self.polygons.copy())
    def apply_to_all(self, func, args):
        for i in range(len(self.polygons)):
            self.polygons[i] = func(self.polygons[i], *args)
        return self
    def apply_pos(self, pos):
        delta = pos - self.polygons[0].verts[0]
        for i in range(len(self.polygons)):
            for j in range(3):
                self.polygons[i].verts[j] = self.polygons[i].verts[j] + delta
        return self
    def __add__(A, B):
        return mesh(A.polygons.copy() + B.polygons.copy())

def grad_func(k):
    return 1 - (1 - k) ** (1 / 1.1)

def mix(col, COL, k):
    return [int(col[i] * grad_func(k) + COL[i] * (1 - grad_func(k))) for i in range(3)]

class polygon_optimised:
    inds = [0, 0, 0]
    color = [250, 100, 100]
    def __init__(self, inds, col = [0, 100, 0]):
        self.color = col.copy()
        self.inds = inds.copy()
    def dist(self, cam, dists):
        return sum([dists[x] for x in self.inds]) / 3
    def draw(self, cam, drops, scr, dists):
        answ = [drops[self.inds[0]], drops[self.inds[1]], drops[self.inds[2]]]
        A = vertex(*answ[0])
        B = vertex(*answ[1])
        C = vertex(*answ[2])
        if vecmul(B - A, C - A) < 0 and any(x.pos[2] >= 0 for x in (A, B, C)):
            #pygame.draw.polygon(scr, mix(SKYCOL, self.color, min(self.dist(cam, dists) / DRAWDIST, 1)),
            #                    [(MOD(int(p[0] + cam.resolution[0] // 2), 10000), MOD(int(p[1] + cam.resolution[1] // 2), 10000)) for p in answ])
            pygame.draw.polygon(scr, self.color,
                                [(MOD(int(p[0] + cam.resolution[0] // 2), 10000), MOD(int(p[1] + cam.resolution[1] // 2), 10000)) for p in answ])
    def __lt__(A, B):
        return True

class mesh_optimised:
    verts = dict()
    polygons = []
    dirrection = [0, 0, 0]
    s_polys = []
    def remove_vertex(self, ind):
        self.verts.pop(ind)
    def weld(self, I, J, usg={}):
        I, J = min(I, J), max(I, J)
        for i in usg[J]:
            pol = self.polygons[i]
            for ind, IND in enumerate(pol.inds):
                if IND == J:
                    usg[J].pop(usg[J].index(i))
                    self.polygons[i].inds[ind] = I
                    try:
                        usg[I].push_back(i)
                    except KeyError:
                        usg[I] = [i]
        self.verts.pop(J)
    def optimise(self):
        print('Optimising')
        print('.' + '_' * 51 + '.')
        counter = 0
        cur = 0
        print(end = ' ')
        n = len(self.verts)
        usage = dict()
        for i in range(len(self.polygons)):
            pol = self.polygons[i]
            for ind, IND in enumerate(pol.inds):
                try:
                    usage[IND].push_back(ind)
                except KeyError:
                    usage[IND] = [ind]
        #print(usage)
        for i in list(self.verts.keys())[::-1]:
            for j in list(self.verts.keys())[::-1]:
                if i != j and dist(self.verts[i], self.verts[j]) <= 0.00001 and False:
                    #self.weld(i, j, usage)
                    pass
                counter += 1
                if 50 * (counter / (n ** 2)) > cur:
                    print(end = 'H')
                    cur += 1
        print()
    def __init__(self, polygons):
        cnt = 0
        for pol in polygons:
            cnt += 1
            if cnt % 10000 == 0 and False:
                print(cnt, len(polygons))
            L = cnt * 3
            for i in range(3):
                self.verts[L + 1 + i] = pol.verts[i]
            self.polygons.append(polygon_optimised([L + 1, L + 2, L + 3], pol.color.copy()))
        #self.optimise()
    def sort(self):
        answ = self.verts.copy()
        for i in answ.keys():
            answ[i] = drop(answ[i], cam)
        dists = self.verts.copy()
        for i in dists.keys():
            dists[i] = dist(dists[i], cam.pos)
        #print(dists)
        self.s_polys = sorted([[p.dist(cam, dists), p] for p in self.polygons])[::-1]
        self.answ = answ
        self.dists = dists
    def draw(self, cam, picture, pointer, step):
        for D, pol in self.s_polys[min(len(self.polygons), pointer):min(len(self.polygons), pointer + step)]:
            pol.draw(cam, self.answ, picture, self.dists)
    def rotate(self, rot, orig = vertex()):
        for i in range(len(self.verts)):
            self.verts[i] = self.verts[i].rotated_origin(orig, rot)
        self.dirrection = [self.dirrection[i] + rot[i] for i in range(3)]
    def set_rot(self, rot, orig = vertex()):
        drot = [rot[i] - self.dirrection[i] for i in range(3)]
        self.rotate(drot, orig)
        self.dirrection = rot.copy()
        return self
    def copy(self):
        return mesh(self.polygons.copy())
    def apply_to_all(self, func, args):
        for i in self.verts.keys():
            self.verts[i] = func(self.verts[i], *args)
        return self
    def apply_pos(self, pos):
        delta = pos - self.verts[0]
        for i in self.verts.keys():
            self.verts[i] = self.verts[i] + delta
        return self
    def find(self, vert, delta = 0.001):
        for i, v in self.verts.items():
            if dist(v, vert) <= delta:
                return i
        return -1
    def add(self, pol):
        binds = [self.find(p) for p in pol.verts]
        for i, bind in enumerate(binds):
            if bind == -1:
                binds[i] = max(self.verts.keys()) + 1
                self.verts[binds[i]] = pol.verts[i]
        self.polygons.append(polygon_optimised(binds, pol.color))
    def remove(self, pol):
        binds = sorted([self.find(p) for p in pol.verts])
        for i, bind in enumerate(binds):
            if bind == -1:
                return False
        rem = True
        for i, poly in list(enumerate(self.polygons))[::-1]:
            if sorted(poly.inds) == binds:
                self.polygons.pop(i)
            elif any([(ind in binds) for ind in poly.inds]):
                rem = False
        if rem:
            self.remove_vertex()
    def __add__(A, B):
        return mesh_optimised(A.polygons.copy() + B.polygons.copy())

class cube:
    top = mesh([])
    bottom = mesh([])
    left = mesh([])
    right = mesh([])
    face = mesh([])
    back = mesh([])
    ALL = mesh([])
    def __init__(self, pos, side, cols):
        col = cols.copy()
        up = vertex(0, 0, 1) * side
        down = vertex(0, 0, -1) * side
        left = vertex(0, -1, 0) * side
        right = vertex(0, 1, 0) * side
        front = vertex(1, 0, 0) * side
        back = vertex(-1, 0, 0) * side
        while len(col) < 6:
            if len(col) > 0:
                col.append(col[-1])
            else:
                col.append([255, 0, 0])
        self.top = mesh([polygon([pos + up, pos + up + right, pos + up + right + front], col[0]),#top
                polygon([pos + up, pos + up + right + front, pos + up + front], col[0])])
        self.bottom = mesh([polygon([pos, pos + right + front, pos + right], col[1]),#bottom
                polygon([pos, pos + front, pos + right + front], col[1])])
        self.left = mesh([polygon([pos, pos + up, pos + up + front], col[4]),#left
                polygon([pos, pos + up + front, pos + front], col[4])])
        self.face = mesh([polygon([pos, pos + up + right, pos + up], col[2]),#face
                polygon([pos, pos + right, pos + up + right], col[2])])
        self.right = mesh([polygon([pos + right, pos + up + right + front, pos + up + right], col[5]),#right
                polygon([pos + right, pos + right + front, pos + up + right + front], col[5])])
        self.back = mesh([polygon([pos + front, pos + up + front, pos + right + front], col[3]),#back
                polygon([pos + front + up, pos + up + right + front, pos + right + front], col[3])])
        self.ALL = self.top + self.bottom + self.left + self.right + self.face + self.back

GRASS = [100, [[0, 100, 0], [70, 50, 0]]]
DIRT = [100, [[60, 40, 0]]]
STONE = [100, [[100, 100, 100]]]
MATERIALS = [GRASS, DIRT, STONE]

def make_cube(side, pos, cols):
    col = cols.copy()
    up = vertex(0, 0, 1) * side
    down = vertex(0, 0, -1) * side
    left = vertex(0, -1, 0) * side
    right = vertex(0, 1, 0) * side
    front = vertex(1, 0, 0) * side
    back = vertex(-1, 0, 0) * side
    while len(col) < 6:
        if len(col) > 0:
            col.append(col[-1])
        else:
            col.append([255, 0, 0])
    answ = [polygon([pos + up, pos + up + right, pos + up + right + front], col[0]),#top
            polygon([pos + up, pos + up + right + front, pos + up + front], col[0]),
            polygon([pos, pos + right + front, pos + right], col[1]),#bottom
            polygon([pos, pos + front, pos + right + front], col[1]),
            polygon([pos, pos + up, pos + up + front], col[4]),#left
            polygon([pos, pos + up + front, pos + front], col[4]),
            polygon([pos, pos + up + right, pos + up], col[2]),#face
            polygon([pos, pos + right, pos + up + right], col[2]),
            polygon([pos + right, pos + up + right + front, pos + up + right], col[5]),#right
            polygon([pos + right, pos + right + front, pos + up + right + front], col[5]),
            polygon([pos + front, pos + up + front, pos + right + front], col[3]),#back
            polygon([pos + front + up, pos + up + right + front, pos + right + front], col[3])]
    return mesh(answ)

class generator:
    def normalise(self, x, m):
        dx = x % m
        if (x // m) % 2 == 0:
            return dx
        else:
            return m - dx
    def not_perlin(self, x, y, h, min_h = 0, zoom = 1 / 5):
        t = 0
        x = x * zoom
        y = y * zoom
        n = sin(x - t * 5) * cos(y) * self.normalise(x + t / 2, 1) + sin(x * 1.1 + y + t * 5) * cos(y / 2 + t * 5) * sin(self.normalise(t, 1) + 1) + sin(y * 0.9 - t * 5 - x * 1.2) * cos(t) * self.normalise(y, 1)
        n /= 4
        if n < 0.9 and False:
            n = 0
        return int(n * h + min_h)

def not_perlin(x, y, h, min_h = 0, zoom = 1 / 40):
        t = 0
        x = x * zoom
        y = y * zoom
        n = sin(x - t * 5) * cos(y) + sin(x * 1.1 + y + t * 5) * cos(y / 2 + t * 5) * sin(t + 1) + sin(y * 0.9 - t * 5 - x * 1.2) * cos(t)
        n /= 3
        if n < 0.9 and False:
            n = 0
        return int(n * h + min_h) % (min_h + h)

def srf(a, b, c):
    dA = dist(a, b)
    dB = dist(b, c)
    dC = dist(c, a)
    s = (dA + dB + dC) / 2
    return sqrt(s * (s - dA) * (s - dB) * (s - dC))

class tile:
    side = 100
    tile_mesh = mesh([])
    grid = []
    min_h = 15
    delta_h = 5
    def __init__(self, name):
        multip = 10
        pic = pygame.image.load(name)
        vmul = multip * min(pic.get_width(), pic.get_height()) / 5
        sx = pic.get_width() // 2
        sy = pic.get_height() // 2
        cnt = 0
        for i in range(pic.get_width() - 1):
            for j in range(pic.get_height() - 1):
                cnt += 1
                if cnt % 1000 == 0:
                    print(cnt, (pic.get_width() - 1) * (pic.get_height() - 1))
                me = vertex((i - sx) * multip, (j - sy) * multip, pic.get_at([i, j])[0] * vmul // 255)
                mep = vertex(me.X(), me.Y(), 0)
                up = vertex((i - sx) * multip, ((j - sy) + 1) * multip, pic.get_at([i, j + 1])[0] * vmul // 255)
                upp = vertex(up.X(), up.Y(), 0)
                right = vertex(((i - sx) + 1) * multip, (j - sy) * multip, pic.get_at([i + 1, j])[0] * vmul // 255)
                rightp = vertex(right.X(), right.Y(), 0)
                dr = vertex(((i - sx) + 1) * multip, ((j - sy) + 1) * multip, pic.get_at([i + 1, j + 1])[0] * vmul // 255)
                drp = vertex(dr.X(), dr.Y(), 0)
                self.tile_mesh.polygons.append(polygon([me, up, dr], [int(255 * srf(mep, upp, drp) / srf(me, up, dr))] * 3))
                self.tile_mesh.polygons.append(polygon([me, dr, right], [int(255 * srf(mep, drp, rightp) / srf(me, dr, right))] * 3))
        print('rewriting')
        self.tile_mesh = mesh_optimised(self.tile_mesh.polygons)
        print(len(self.tile_mesh.verts))

SMUL = 1000

v = vertex(1, 0, 1)
rot = [0, PI / 4, PI / 4]
print(v.get_rotation())
print(v.apply_rot_y(rot[1]).pos)
print(sorted([0, 1, 2, 3]))
cam = camera()
pol = polygon([vertex(700, 200, 250), vertex(700, -200, 250), vertex(700, 20, -50)])
#cube = make_cube(100, vertex(650, -50, -50), [[0, 100, 0], [70, 50, 0]])
NAME = 'Result.bmp'
STEP = 10
print('Generating tile...')
tl = tile(NAME)
scr = pygame.display.set_mode(cam.resolution)
kg = True
speed = vertex()
tm = time.monotonic()
mpos = vertex()
log = open('log.txt', 'w')
pointer = 0
step = 10
print('Sorting...')
tl.tile_mesh.sort()
pic = pygame.image.load(NAME)
OK = True
print('ready')
while kg:
    TM = time.monotonic()
    delta_time = TM - tm
    tm = TM
    log.write(str(delta_time) + '\n')
    if pygame.mouse.get_pressed()[0]:
        pygame.mouse.set_visible(False)
        rel = list(pygame.mouse.get_rel())
        rel[1] *= -1
        mrel = vertex(*([0] + rel[::-1]))
    else:
        pygame.mouse.set_visible(True)
        mrel = vertex()
        pygame.mouse.get_rel()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            kg = False
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_w, pygame.K_UP]:
                speed = speed + vertex(0, 1)
                #cam.rot = (vertex(*cam.rot) + speed * 0.1).pos
            if event.key in [pygame.K_s, pygame.K_DOWN]:
                speed = speed + vertex(0, -1)
                #cam.rot = (vertex(*cam.rot) + speed * 0.1).pos
            if event.key in [pygame.K_a, pygame.K_LEFT]:
                speed = speed + vertex(0, 0, -1)
                #cam.rot = (vertex(*cam.rot) + speed * 0.1).pos
            if event.key in [pygame.K_d, pygame.K_RIGHT]:
                speed = speed + vertex(0, 0, 1)
                #cam.rot = (vertex(*cam.rot) + speed * 0.1).pos
            if event.key in [pygame.K_ESCAPE]:
                kg = False
            if event.key in [pygame.K_b]:
                tl.set_at(vertex(len(tl.grid) - 3, len(tl.grid) - 3, len(tl.grid[0][0]) - 5), 1)
        if event.type == pygame.KEYUP:
            if event.key in [pygame.K_w, pygame.K_UP]:
                speed = speed - vertex(0, 1)
            if event.key in [pygame.K_s, pygame.K_DOWN]:
                speed = speed - vertex(0, -1)
            if event.key in [pygame.K_a, pygame.K_LEFT]:
                speed = speed - vertex(0, 0, -1)
            if event.key in [pygame.K_d, pygame.K_RIGHT]:
                speed = speed - vertex(0, 0, 1)
    #cam.rot = (vertex(*cam.rot) + mrel * delta_time * 40 / cam.resolution[0]).pos
    cam.rot = (vertex(*cam.rot) + speed * 0.01).pos
    length = max(pic.get_height(), pic.get_width()) * STEP * 2
    cam.pos = vertex(-cos(cam.rot[2]) * cos(cam.rot[1]) * length, -sin(cam.rot[2]) * cos(cam.rot[1]) * length, sin(-cam.rot[1]) * length)
    try:
        if dist(speed) != 0:
            OK = True
            scr.fill(SKYCOL)
            tl.tile_mesh.draw(cam, scr, 0, step)
            #print(0)
            #speed = vertex(0)
        if dist(speed) == 0 and OK:
            OK = False
            pointer = 0
            scr.fill(SKYCOL)
            print('sorting')
            tl.tile_mesh.sort()
    except ZeroDivisionError:
        pass
    #cam.pos = cam.pos + (speed * delta_time * SMUL).rotated(cam.rot)
    #scr.fill(SKYCOL)
    #cube.apply_to_all(polygon.apply_to_all, [vertex.add_rot_y, [0]])
    #cube.rotate([0, 0.5 * delta_time, 1 * delta_time], vertex(700, 0, 0))
    #cb.set_rot([0, (mpos[1] - cam.resolution[1]) / 200, (mpos[0] - cam.resolution[0]) / 200], vertex(700, 0, 0))
    #cube.set_rot([0, 0, (mpos[0] - cam.resolution[0]) / 200], vertex(700, 0, 0))
    #res = cube.draw(cam)
    tl.tile_mesh.draw(cam, scr, pointer, step)
    pointer += step
    pygame.display.update()
    #print(pointer)
pygame.quit()
log.close()
