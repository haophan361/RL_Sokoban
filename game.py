import pygame
import assets
from objects.box import Box
from objects.box_docked import BoxDocked
from objects.dock import Dock
from objects.floor import Floor
from objects.wall import Wall
from objects.worker import Worker

class Game:
    def __init__(self, matrix):
        self.matrix = [list(row) for row in matrix] if matrix else []

    def set_matrix(self, matrix):
        self.matrix = [list(row) for row in matrix] if matrix else []

    def get_dimensions(self):
        if not self.matrix:
            return 0, 0
        h = len(self.matrix)
        w = max(len(row) for row in self.matrix)
        return w, h

    def load_size(self, tile_size=64):
        w, h = self.get_dimensions()
        return w * tile_size, h * tile_size

    def print_game(self, screen, offset_x=0, offset_y=0, tile_size=64):
        if not self.matrix:
            return

        object_map = {
            '#': Wall,
            '@': Worker,
            '.': Dock,
            '$': Box,
            '*': BoxDocked,
        }

        for r_idx, row in enumerate(self.matrix):
            for c_idx, char in enumerate(row):
                px = offset_x + c_idx * tile_size
                py = offset_y + r_idx * tile_size

                if char in object_map:
                    obj = object_map[char](px, py)
                    screen.blit(obj.image, obj.rect)

    @staticmethod
    def fill_screen_with_floor(size, screen, tile_size=64):
        screen_width, screen_height = size
        for x in range(0, screen_width, tile_size):
            for y in range(0, screen_height, tile_size):
                floor = Floor(x, y)
                screen.blit(floor.image, floor.rect)
