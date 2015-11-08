import pygame
import pygame.gfxdraw
import numpy as np

class Colors(object):
    black = (0, 0, 0)
    white = (255, 255, 255)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)

class Viewer2D(object):

    def __init__(self, size=(640, 480), xlim=None, ylim=None):
        pygame.init()
        screen = pygame.display.set_mode(size)
        if xlim is None:
            xlim = (0, self.size[0])
        if ylim is None:
            ylim = (0, self.size[1])
        self._screen = screen
        self._xlim = xlim
        self._ylim = ylim

    @property
    def xlim(self):
        return self._xlim

    @xlim.setter
    def xlim(self, value):
        self._xlim = value

    @property
    def ylim(self):
        return self._ylim

    @ylim.setter
    def ylim(self, value):
        self._ylim = value

    def reset(self):
        self.fill(Colors.white)

    def fill(self, color):
        self.screen.fill(color)

    def scale_x(self, world_x):
        xmin, xmax = self.xlim
        return int((world_x - xmin) * self.screen.get_width() / (xmax - xmin))

    def scale_y(self, world_y):
        ymin, ymax = self.ylim
        return int((self.screen.get_height() - (world_y - ymin) * self.screen.get_height() / (ymax - ymin)))

    def scale_point(self, point):
        x, y = point
        return (self.scale_x(x), self.scale_y(y))

    @property
    def scale_factor(self):
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        return min(self.screen.get_width() / (xmax - xmin), self.screen.get_height() / (ymax - ymin))

    def scale_size(self, size):
        return size * self.scale_factor

    def line(self, p1, p2, color, width=None):
        if width is None:
            width = 1
        else:
            width = int(width * self.scale_factor)
        x1, y1 = self.scale_point(p1)
        x2, y2 = self.scale_point(p2)
        pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)

    def circle(self, p, radius, color):
        pygame.draw.circle(self.screen, color, self.scale_point(p), int(self.scale_size(radius)))

    @property
    def screen(self):
        return self._screen

    def loop_once(self):
        pygame.display.flip()

    # Draw a checker background
    def checker(self, colors=[Colors.white, Colors.black], granularity=4, offset=(0, 0)):
        screen_height = self.screen.get_height()
        screen_width = self.screen.get_width()
        screen_size = min(screen_height, screen_width)
        checker_size = int(screen_size / granularity)
        offset_x = self.scale_x(offset[0] + self.xlim[0])
        offset_y = self.scale_y(offset[1] + self.ylim[0])
        start_idx = int(offset_x / checker_size) + int(offset_y / checker_size)
        offset_x = ((offset_x % checker_size) + checker_size) % checker_size
        offset_y = ((offset_y % checker_size) + checker_size) % checker_size
        for row in range(-1, int(np.ceil(screen_height * 1.0 / checker_size))+1):
            for col in range(-1, int(np.ceil(screen_width * 1.0 / checker_size))+1):
                the_square = (col*checker_size+offset_x, row*checker_size+offset_y, checker_size, checker_size)
                self.screen.fill(colors[(start_idx+row+col)%2], the_square)
