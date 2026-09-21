"""Interactive pygame maze-search visualizer.

Algorithms: BFS, DFS, Dijkstra, Greedy Best-First, and A*.
Contributor: Codex.
Run: pip install pygame && python maze_search-3.py
"""

import sys
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

import pygame

DEFAULT_GRID_SIZE = 20
CELL_SIZE = 28
MAX_GRID_PIXELS = 640
SIDEBAR_WIDTH = 250
TOP_BAR_HEIGHT = 80
MIN_WINDOW_WIDTH = 940

FPS = 60

SPEED_PRESETS = {
    "Slow": (1, 5),
    "Normal": (2, 2),
    "Fast": (8, 1),
}
SPEED_ORDER = ["Slow", "Normal", "Fast"]

COLOR_BG = (30, 33, 41)
COLOR_TOPBAR = (24, 26, 33)
COLOR_SIDEBAR = (44, 48, 60)
COLOR_GRID_LINE = (90, 94, 104)
COLOR_EMPTY = (245, 246, 248)
COLOR_WALL = (35, 37, 43)
COLOR_START = (46, 204, 113)
COLOR_END = (231, 76, 60)
COLOR_FRONTIER = (241, 196, 15)
COLOR_PATH = (52, 152, 219)
COLOR_TEXT = (236, 240, 241)
COLOR_TEXT_DIM = (160, 165, 175)
COLOR_BTN = (60, 65, 80)
COLOR_BTN_HOVER = (75, 80, 98)
COLOR_BTN_ACTIVE = (41, 128, 185)
COLOR_BTN_RUN = (39, 174, 96)
COLOR_BTN_DANGER = (192, 57, 43)


@dataclass
class Node:
    """One candidate path to a tile in the search fringe."""

    cell: tuple[int, int]
    parent: "Node | None"
    distance: int
    heuristic: int


def neighbors(cell, walls, grid_size):
    """Yield traversable orthogonal neighbors."""
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in walls:
            yield (nr, nc)


def reconstruct_path(parent, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def manhattan(cell, end):
    return abs(cell[0] - end[0]) + abs(cell[1] - end[1])


class IndexedMinHeap:
    """Min-heap with one entry per cell and decrease-key support."""

    def __init__(self):
        self.heap = []
        self.positions = {}

    def __len__(self):
        return len(self.heap)

    def __contains__(self, cell):
        return cell in self.positions

    def push_or_decrease(self, node, node_priority):
        cell = node.cell
        if cell in self.positions:
            index = self.positions[cell]
            if node_priority >= self.heap[index][0]:
                return False
            self.heap[index] = [node_priority, node]
            self._sift_up(index)
            return False
        self.heap.append([node_priority, node])
        self.positions[cell] = len(self.heap) - 1
        self._sift_up(len(self.heap) - 1)
        return True

    def pop_min(self):
        _, node = self.heap[0]
        last = self.heap.pop()
        del self.positions[node.cell]
        if self.heap:
            self.heap[0] = last
            self.positions[last[1].cell] = 0
            self._sift_down(0)
        return node

    def _swap(self, left, right):
        self.heap[left], self.heap[right] = self.heap[right], self.heap[left]
        self.positions[self.heap[left][1].cell] = left
        self.positions[self.heap[right][1].cell] = right

    def _sift_up(self, index):
        while index:
            parent = (index - 1) // 2
            if self.heap[parent][0] <= self.heap[index][0]:
                return
            self._swap(parent, index)
            index = parent

    def _sift_down(self, index):
        while True:
            left, right, smallest = 2 * index + 1, 2 * index + 2, index
            if left < len(self.heap) and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < len(self.heap) and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == index:
                return
            self._swap(index, smallest)
            index = smallest


def search(start, end, walls, grid_size, algorithm, dedup_mode):
    """Run one search, yielding fringe additions and returning its result."""
    frontier = deque() if algorithm == "BFS" else []
    priority_frontier = IndexedMinHeap() if algorithm not in ("BFS", "DFS") and dedup_mode == "push" else None
    serial = count()
    discovered, visited, parents = set(), set(), {}
    fringe_count = max_fringe_size = 0

    def priority(node):
        if algorithm == "Dijkstra":
            return node.distance
        if algorithm == "Greedy":
            return node.heuristic
        if algorithm == "A*":
            return node.distance + node.heuristic
        return 0

    def add(node):
        nonlocal fringe_count, max_fringe_size
        if algorithm in ("BFS", "DFS"):
            frontier.append(node)
        elif priority_frontier is not None:
            priority_frontier.push_or_decrease(node, priority(node))
        else:
            heappush(frontier, (priority(node), next(serial), node))
        fringe_count += 1
        max_fringe_size = max(max_fringe_size, len(priority_frontier or frontier))

    def take():
        if algorithm == "BFS":
            return frontier.popleft()
        if algorithm == "DFS":
            return frontier.pop()
        if priority_frontier is not None:
            return priority_frontier.pop_min()
        return heappop(frontier)[2]

    start_node = Node(start, None, 0, manhattan(start, end))
    add(start_node)
    discovered.add(start)
    yield ("frontier", start, len(priority_frontier or frontier))

    while len(priority_frontier) > 0 if priority_frontier is not None else frontier:
        node = take()
        if node.cell in visited:
            continue
        visited.add(node.cell)
        if node.parent is not None:
            parents[node.cell] = node.parent.cell
        if node.cell == end:
            return {"found": True, "path": reconstruct_path(parents, start, end),
                    "fringe_count": fringe_count, "max_fringe_size": max_fringe_size}
        for cell in neighbors(node.cell, walls, grid_size):
            if cell in visited:
                continue
            child = Node(cell, node, node.distance + 1, manhattan(cell, end))
            if dedup_mode == "push":
                if cell in discovered:
                    if priority_frontier is not None and cell in priority_frontier:
                        priority_frontier.push_or_decrease(child, priority(child))
                    continue
                discovered.add(cell)
            add(child)
            yield ("frontier", cell, len(priority_frontier or frontier))

    return {"found": False, "path": None, "fringe_count": fringe_count,
            "max_fringe_size": max_fringe_size}


def bfs(start, end, walls, grid_size, dedup_mode):
    return (yield from search(start, end, walls, grid_size, "BFS", dedup_mode))


def dfs(start, end, walls, grid_size, dedup_mode):
    return (yield from search(start, end, walls, grid_size, "DFS", dedup_mode))


def iterative_deepening_dfs(start, end, walls, grid_size, dedup_mode):
    """Repeat depth-limited DFS, raising the limit after each complete pass."""
    fringe_count = max_fringe_size = 0
    depth_limit = 0

    while True:
        stack = [Node(start, None, 0, manhattan(start, end))]
        best_depth, parents = ({start: 0} if dedup_mode == "push" else {}), {}
        reached_limit = False
        yield ("iteration", depth_limit, len(stack))
        fringe_count += 1
        max_fringe_size = max(max_fringe_size, len(stack))
        yield ("frontier", start, len(stack))

        while stack:
            node = stack.pop()
            if dedup_mode == "pop":
                known_depth = best_depth.get(node.cell)
                if known_depth is not None and node.distance >= known_depth:
                    continue
                best_depth[node.cell] = node.distance
            if node.parent is not None:
                parents[node.cell] = node.parent.cell
            if node.cell == end:
                return {"found": True, "path": reconstruct_path(parents, start, end),
                        "fringe_count": fringe_count, "max_fringe_size": max_fringe_size}
            if node.distance == depth_limit:
                reached_limit = True
                continue
            for cell in neighbors(node.cell, walls, grid_size):
                child = Node(cell, node, node.distance + 1, manhattan(cell, end))
                if dedup_mode == "push":
                    if child.distance >= best_depth.get(cell, float("inf")):
                        continue
                    best_depth[cell] = child.distance
                stack.append(child)
                fringe_count += 1
                max_fringe_size = max(max_fringe_size, len(stack))
                yield ("frontier", cell, len(stack))

        if not reached_limit:
            return {"found": False, "path": None, "fringe_count": fringe_count,
                    "max_fringe_size": max_fringe_size}
        depth_limit += 1


def dijkstra(start, end, walls, grid_size, dedup_mode):
    return (yield from search(start, end, walls, grid_size, "Dijkstra", dedup_mode))


def greedy_best_first(start, end, walls, grid_size, dedup_mode):
    return (yield from search(start, end, walls, grid_size, "Greedy", dedup_mode))


def a_star(start, end, walls, grid_size, dedup_mode):
    return (yield from search(start, end, walls, grid_size, "A*", dedup_mode))


ALGORITHMS = {
    "BFS": bfs,
    "DFS": dfs,
    "IDDFS": iterative_deepening_dfs,
    "Dijkstra": dijkstra,
    "Greedy": greedy_best_first,
    "A*": a_star,
}



class Button:
    def __init__(self, rect, label, kind="normal"):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.kind = kind
        self.active = False

    def draw(self, surface, font, mouse_pos):
        hovered = self.rect.collidepoint(mouse_pos)
        if self.kind == "run":
            color = COLOR_BTN_RUN
        elif self.kind == "danger":
            color = COLOR_BTN_DANGER
        elif self.active:
            color = COLOR_BTN_ACTIVE
        elif hovered:
            color = COLOR_BTN_HOVER
        else:
            color = COLOR_BTN

        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_GRID_LINE, self.rect, width=1, border_radius=6)
        text_surf = font.render(self.label, True, COLOR_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


class MazeApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Maze Search Visualizer")

        grid_size = DEFAULT_GRID_SIZE
        start = (2, 2)
        end = (DEFAULT_GRID_SIZE - 3, DEFAULT_GRID_SIZE - 3)
        walls = set()

        self.grid_size = grid_size
        self.cell_size = max(8, min(CELL_SIZE, MAX_GRID_PIXELS // grid_size))
        self.grid_pixels = self.grid_size * self.cell_size
        self.window_width = max(self.grid_pixels + SIDEBAR_WIDTH, MIN_WINDOW_WIDTH)
        self.sidebar_x = self.window_width - SIDEBAR_WIDTH

        self.mode = "wall"
        self.algorithm = "BFS"
        self.dedup_mode = "push"
        self.speed = "Normal"

        self._build_buttons()
        self.window_height = max(self.grid_pixels + TOP_BAR_HEIGHT, self._min_sidebar_height)

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("segoeui", 16)
        self.font_bold = pygame.font.SysFont("segoeui", 18, bold=True)
        self.font_small = pygame.font.SysFont("segoeui", 13)

        self.walls = walls
        self.start = start
        self.end = end

        self.state = "draw"
        self.generator = None
        self.frontier_set = set()
        self.live_fringe_count = 0
        self.live_max_fringe = 0
        self.result = None
        self.path_reveal_index = 0
        self.frame_counter = 0
        self.message = ""

        self.dragging = False
        self.drag_paint_wall = True

    def _build_buttons(self):
        x = self.sidebar_x + 20
        w = SIDEBAR_WIDTH - 40
        h = 34
        gap = 8
        y = TOP_BAR_HEIGHT + 16

        self.btn_mode_wall = Button((x, y, w, h), "Draw: Walls")
        y += h + gap
        self.btn_mode_start = Button((x, y, w, h), "Draw: Start Tile")
        y += h + gap
        self.btn_mode_end = Button((x, y, w, h), "Draw: End Tile")
        y += h + gap + 12

        self.btn_algo_bfs = Button((x, y, w, h), "BFS")
        y += h + gap
        self.btn_algo_dfs = Button((x, y, w, h), "DFS")
        y += h + gap
        self.btn_algo_iddfs = Button((x, y, w, h), "Iterative Deepening DFS")
        y += h + gap
        self.btn_algo_dijkstra = Button((x, y, w, h), "Dijkstra (UCS)")
        y += h + gap
        self.btn_algo_greedy = Button((x, y, w, h), "Greedy Best-First")
        y += h + gap
        self.btn_algo_astar = Button((x, y, w, h), "A*")
        y += h + gap + 12

        self.btn_dedup = Button((x, y, w, h), "Dedup: Push")
        y += h + gap
        self.btn_speed = Button((x, y, w, h), "Speed: Normal")
        y += h + gap + 12

        self.btn_run = Button((x, y, w, h + 6), "Run Search", kind="run")
        y += h + 6 + gap
        self.btn_clear_path = Button((x, y, w, h), "Clear Path")
        y += h + gap
        self.btn_reset = Button((x, y, w, h), "Reset Grid", kind="danger")
        y += h + gap

        self.mode_buttons = {
            "wall": self.btn_mode_wall,
            "start": self.btn_mode_start,
            "end": self.btn_mode_end,
        }
        self.algo_buttons = {
            "BFS": self.btn_algo_bfs,
            "DFS": self.btn_algo_dfs,
            "IDDFS": self.btn_algo_iddfs,
            "Dijkstra": self.btn_algo_dijkstra,
            "Greedy": self.btn_algo_greedy,
            "A*": self.btn_algo_astar,
        }
        self._sync_toggle_states()
        self._instructions_y = y + 8
        self._min_sidebar_height = self._instructions_y + 210

    def _sync_toggle_states(self):
        for key, btn in self.mode_buttons.items():
            btn.active = (key == self.mode)
        for key, btn in self.algo_buttons.items():
            btn.active = (key == self.algorithm)

    def pixel_to_cell(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_pixels or y < TOP_BAR_HEIGHT or y >= TOP_BAR_HEIGHT + self.grid_pixels:
            return None
        col = x // self.cell_size
        row = (y - TOP_BAR_HEIGHT) // self.cell_size
        return (row, col)

    def is_editable(self):
        return self.state in ("draw", "done", "no_path")

    def reset_visualization(self):
        self.state = "draw"
        self.generator = None
        self.frontier_set = set()
        self.live_fringe_count = 0
        self.live_max_fringe = 0
        self.result = None
        self.path_reveal_index = 0
        self.message = ""

    def reset_grid(self):
        self.walls.clear()
        self.start = (2, 2)
        self.end = (self.grid_size - 3, self.grid_size - 3)
        self.reset_visualization()

    def handle_cell_edit(self, cell, is_initial_click):
        if not self.is_editable():
            return
        if cell is None:
            return
        if self.state in ("done", "no_path"):
            self.reset_visualization()

        if self.mode == "wall":
            if cell == self.start or cell == self.end:
                return
            if is_initial_click:
                self.drag_paint_wall = cell not in self.walls
            if self.drag_paint_wall:
                self.walls.add(cell)
            else:
                self.walls.discard(cell)
        elif self.mode == "start":
            if cell != self.end:
                self.walls.discard(cell)
                self.start = cell
        elif self.mode == "end":
            if cell != self.start:
                self.walls.discard(cell)
                self.end = cell

    def start_search(self):
        if self.state in ("searching", "path_animating"):
            return
        if self.start is None or self.end is None:
            self.message = "Set both a start and end tile first."
            return
        self.reset_visualization()
        algo_fn = ALGORITHMS[self.algorithm]
        self.generator = algo_fn(self.start, self.end, self.walls, self.grid_size, self.dedup_mode)
        self.state = "searching"
        self.frame_counter = 0

    def handle_button_click(self, pos):
        if self.btn_mode_wall.clicked(pos):
            self.mode = "wall"
        elif self.btn_mode_start.clicked(pos):
            self.mode = "start"
        elif self.btn_mode_end.clicked(pos):
            self.mode = "end"
        elif self.btn_algo_bfs.clicked(pos):
            self.algorithm = "BFS"
        elif self.btn_algo_dfs.clicked(pos):
            self.algorithm = "DFS"
        elif self.btn_algo_iddfs.clicked(pos):
            self.algorithm = "IDDFS"
        elif self.btn_algo_dijkstra.clicked(pos):
            self.algorithm = "Dijkstra"
        elif self.btn_algo_greedy.clicked(pos):
            self.algorithm = "Greedy"
        elif self.btn_algo_astar.clicked(pos):
            self.algorithm = "A*"
        elif self.btn_dedup.clicked(pos):
            self.dedup_mode = "pop" if self.dedup_mode == "push" else "push"
            self.btn_dedup.label = f"Dedup: {self.dedup_mode.title()}"
        elif self.btn_speed.clicked(pos):
            idx = SPEED_ORDER.index(self.speed)
            self.speed = SPEED_ORDER[(idx + 1) % len(SPEED_ORDER)]
            self.btn_speed.label = f"Speed: {self.speed}"
        elif self.btn_run.clicked(pos):
            self.start_search()
        elif self.btn_clear_path.clicked(pos):
            self.reset_visualization()
        elif self.btn_reset.clicked(pos):
            self.reset_grid()
        else:
            return
        self._sync_toggle_states()

    def update_search(self):
        if self.state != "searching":
            return
        self.frame_counter += 1
        steps, frames_between = SPEED_PRESETS[self.speed]
        if self.frame_counter < frames_between:
            return
        self.frame_counter = 0

        for _ in range(steps):
            try:
                kind, cell, fringe_size = next(self.generator)
                if kind == "iteration":
                    self.frontier_set.clear()
                    self.message = f"IDDFS depth limit: {cell}"
                    continue
                if kind == "frontier" and cell not in (self.start, self.end):
                    self.frontier_set.add(cell)
                self.live_fringe_count += 1
                self.live_max_fringe = max(self.live_max_fringe, fringe_size)
            except StopIteration as stop:
                self.result = stop.value
                if self.result and self.result["found"]:
                    self.state = "path_animating"
                    self.path_reveal_index = 0
                else:
                    self.state = "no_path"
                    self.message = "No path exists between start and end."
                return

    def update_path_animation(self):
        if self.state != "path_animating":
            return
        self.frame_counter += 1
        steps, frames_between = SPEED_PRESETS[self.speed]
        if self.frame_counter < frames_between:
            return
        self.frame_counter = 0

        path = self.result["path"]
        for _ in range(max(1, steps // 2)):
            if self.path_reveal_index >= len(path):
                self.state = "done"
                break
            self.path_reveal_index += 1

    def draw_grid(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = (row, col)
                rect = pygame.Rect(
                    col * self.cell_size,
                    TOP_BAR_HEIGHT + row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                if cell == self.start:
                    color = COLOR_START
                elif cell == self.end:
                    color = COLOR_END
                elif cell in self.walls:
                    color = COLOR_WALL
                elif self.state in ("path_animating", "done"):
                    revealed_path = self.result["path"][: self.path_reveal_index] if self.result else []
                    if cell in revealed_path:
                        color = COLOR_PATH
                    elif cell in self.frontier_set:
                        color = COLOR_FRONTIER
                    else:
                        color = COLOR_EMPTY
                elif cell in self.frontier_set:
                    color = COLOR_FRONTIER
                else:
                    color = COLOR_EMPTY

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, width=1)

    def draw_top_bar(self):
        rect = pygame.Rect(0, 0, self.window_width, TOP_BAR_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_TOPBAR, rect)

        fringe_count = self.live_fringe_count
        max_fringe = self.live_max_fringe
        if self.result:
            fringe_count = self.result["fringe_count"]
            max_fringe = self.result["max_fringe_size"]

        path_cost_text = "-"
        if self.result and self.result.get("path") is not None:
            path_cost_text = str(len(self.result["path"]))

        status_text = {
            "draw": "Ready — draw your maze, then Run Search",
            "searching": f"Searching ({self.algorithm})...",
            "path_animating": "Path found! Tracing solution...",
            "done": "Done.",
            "no_path": "No path found.",
        }.get(self.state, "")

        if self.message and self.state in ("draw", "searching"):
            status_text = self.message

        line1 = f"Search cost (fringe adds): {fringe_count}    Max fringe size (memory proxy): {max_fringe}"
        line2 = f"Path cost (tiles): {path_cost_text}"

        surf1 = self.font_bold.render(line1, True, COLOR_TEXT)
        surf2 = self.font_bold.render(line2, True, COLOR_TEXT)
        surf3 = self.font_small.render(status_text, True, COLOR_TEXT_DIM)
        self.screen.blit(surf1, (16, 6))
        self.screen.blit(surf2, (16, 28))
        self.screen.blit(surf3, (16, 56))

    def draw_sidebar(self):
        rect = pygame.Rect(self.sidebar_x, 0, SIDEBAR_WIDTH, self.window_height)
        pygame.draw.rect(self.screen, COLOR_SIDEBAR, rect)

        mouse_pos = pygame.mouse.get_pos()
        for btn in (
            self.btn_mode_wall,
            self.btn_mode_start,
            self.btn_mode_end,
            self.btn_algo_bfs,
            self.btn_algo_dfs,
            self.btn_algo_iddfs,
            self.btn_algo_dijkstra,
            self.btn_algo_greedy,
            self.btn_algo_astar,
            self.btn_dedup,
            self.btn_speed,
            self.btn_run,
            self.btn_clear_path,
            self.btn_reset,
        ):
            btn.draw(self.screen, self.font, mouse_pos)

        instructions = [
            "Tips:",
            "- Drag with mouse to draw",
            "  or erase walls.",
            "- Click a mode, then click",
            "  the grid to place it.",
            "- Dedup Push keeps one fringe",
            "  entry per tile; Pop permits",
            "  duplicates until they pop.",
        ]
        y = self._instructions_y
        for line in instructions:
            surf = self.font_small.render(line, True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (self.sidebar_x + 20, y))
            y += 18

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if pos[0] >= self.sidebar_x:
                        self.handle_button_click(pos)
                    elif pos[0] < self.grid_pixels:
                        cell = self.pixel_to_cell(pos)
                        if cell is not None:
                            self.dragging = True
                            self.handle_cell_edit(cell, is_initial_click=True)

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging = False

                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging and self.mode == "wall":
                        cell = self.pixel_to_cell(event.pos)
                        if cell is not None:
                            self.handle_cell_edit(cell, is_initial_click=False)

            self.update_search()
            self.update_path_animation()

            self.screen.fill(COLOR_BG)
            self.draw_grid()
            self.draw_top_bar()
            self.draw_sidebar()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


def main():
    app = MazeApp()
    app.run()


if __name__ == "__main__":
    main()
