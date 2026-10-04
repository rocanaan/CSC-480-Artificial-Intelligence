"""Tree search demo for BFS, DFS, and iterative deepening DFS.

Run: pip install pygame && python tree_search_demo.py
Contributor: Codex.
"""

from collections import deque
from dataclasses import dataclass
from random import Random
from time import perf_counter

import pygame


WIDTH, HEIGHT = 1040, 680
FPS = 60
BG = (29, 32, 40)
PANEL = (43, 48, 60)
BUTTON = (66, 73, 91)
HOVER = (82, 91, 112)
ACTIVE = (41, 128, 185)
TEXT = (240, 242, 246)
MUTED = (174, 181, 194)
GOOD = (46, 204, 113)
WARN = (241, 196, 15)
BAD = (231, 76, 60)
LINE = (92, 100, 120)


@dataclass(frozen=True)
class TreeSpec:
    branching_factor: int
    max_depth: int
    distribution: str
    fixed_depth: int
    seed: int
    goal_depth: int
    goal_code: int


@dataclass
class Result:
    algorithm: str
    status: str
    expanded: int
    elapsed_ms: float
    max_fringe: int


def choose_goal(branching_factor, max_depth, distribution, fixed_depth, seed):
    rng = Random(seed)
    if distribution == "Fixed depth":
        goal_depth = min(fixed_depth, max_depth)
    else:
        node_index = rng.randrange(sum(branching_factor ** depth for depth in range(max_depth + 1)))
        goal_depth = 0
        while node_index >= branching_factor ** goal_depth:
            node_index -= branching_factor ** goal_depth
            goal_depth += 1
    goal_code = 0
    for _ in range(goal_depth):
        goal_code = goal_code * branching_factor + rng.randrange(branching_factor)
    return goal_depth, goal_code


def make_spec(branching_factor, max_depth, distribution, fixed_depth, seed):
    goal_depth, goal_code = choose_goal(branching_factor, max_depth, distribution, fixed_depth, seed)
    return TreeSpec(branching_factor, max_depth, distribution, fixed_depth, seed, goal_depth, goal_code)


def is_goal(node, spec):
    return node == (spec.goal_depth, spec.goal_code)


def children(node, spec):
    depth, code = node
    if depth == spec.max_depth:
        return ()
    return tuple((depth + 1, code * spec.branching_factor + child) for child in range(spec.branching_factor))


def run_search(algorithm, spec, fringe_cap, time_cap_ms):
    start_time = perf_counter()
    expanded = 0
    max_fringe = 1

    def finish(status):
        return Result(algorithm, status, expanded, (perf_counter() - start_time) * 1000, max_fringe)

    def timed_out():
        return (perf_counter() - start_time) * 1000 >= time_cap_ms

    if fringe_cap < 1:
        return finish("out of memory")

    if algorithm == "IDDFS":
        depth_limit = 0
        fringe = [(0, 0)]
        while True:
            while fringe:
                if timed_out():
                    return finish("timed out")
                node = fringe.pop()
                expanded += 1
                if is_goal(node, spec):
                    return finish("found")
                if node[0] == depth_limit:
                    continue
                new_nodes = children(node, spec)
                if len(fringe) + len(new_nodes) > fringe_cap:
                    return finish("out of memory")
                fringe.extend(reversed(new_nodes))
                max_fringe = max(max_fringe, len(fringe))
            if depth_limit == spec.max_depth:
                return finish("not found")
            depth_limit += 1
            fringe = [(0, 0)]
            max_fringe = max(max_fringe, 1)

    fringe = deque([(0, 0)]) if algorithm == "BFS" else [(0, 0)]
    while fringe:
        if timed_out():
            return finish("timed out")
        node = fringe.popleft() if algorithm == "BFS" else fringe.pop()
        expanded += 1
        if is_goal(node, spec):
            return finish("found")
        if node[0] == spec.max_depth:
            continue
        new_nodes = children(node, spec)
        if len(fringe) + len(new_nodes) > fringe_cap:
            return finish("out of memory")
        if algorithm == "BFS":
            fringe.extend(new_nodes)
        else:
            fringe.extend(reversed(new_nodes))
        max_fringe = max(max_fringe, len(fringe))
    return finish("not found")


class Button:
    def __init__(self, rect, label, action):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.action = action

    def draw(self, surface, font, active=False):
        color = ACTIVE if active else HOVER if self.rect.collidepoint(pygame.mouse.get_pos()) else BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, LINE, self.rect, width=1, border_radius=6)
        rendered = font.render(self.label, True, TEXT)
        surface.blit(rendered, rendered.get_rect(center=self.rect.center))


class TreeSearchApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tree Search Demo")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 17)
        self.bold = pygame.font.SysFont("segoeui", 18, bold=True)
        self.small = pygame.font.SysFont("segoeui", 14)
        self.branching_factor = 2
        self.max_depth = 16
        self.fixed_depth = 12
        self.distribution = "Uniform node"
        self.fringe_cap = 20_000
        self.time_cap_ms = 1_000
        self.seed = Random().randrange(1 << 31)
        self.spec = self.new_spec()
        self.results = {}
        self.message = "Choose an algorithm, or run all three on this same target."
        self.buttons = self.build_buttons()

    def new_spec(self):
        return make_spec(self.branching_factor, self.max_depth, self.distribution, self.fixed_depth, self.seed)

    def regenerate(self):
        self.seed = Random().randrange(1 << 31)
        self.spec = self.new_spec()
        self.results.clear()
        self.message = "New target generated."

    def update_spec(self):
        self.fixed_depth = min(self.fixed_depth, self.max_depth)
        self.spec = self.new_spec()
        self.results.clear()

    def build_buttons(self):
        h = 30
        buttons = []
        actions = (("branch_down", "branch_up"), ("depth_down", "depth_up"),
                   ("fixed_down", "fixed_up"), ("fringe_down", "fringe_up"),
                   ("time_down", "time_up"))
        for row, (down, up) in enumerate(actions):
            y = 107 + row * 43
            buttons.append(Button((270, y, 92, h), "−", down))
            buttons.append(Button((372, y, 92, h), "+", up))
        buttons.extend([
            Button((26, 340, 214, h), "Distribution: Uniform", "distribution"),
            Button((250, 340, 214, h), "New target", "new_target"),
            Button((26, 382, 136, h), "Run BFS", "BFS"),
            Button((172, 382, 136, h), "Run DFS", "DFS"),
            Button((318, 382, 146, h), "Run IDDFS", "IDDFS"),
            Button((26, 424, 438, h), "Run all", "all"),
        ])
        return buttons

    def handle_action(self, action):
        if action == "branch_down":
            self.branching_factor = max(1, self.branching_factor - 1)
        elif action == "branch_up":
            self.branching_factor = min(10, self.branching_factor + 1)
        elif action == "depth_down":
            self.max_depth = max(0, self.max_depth - 1)
        elif action == "depth_up":
            self.max_depth = min(30, self.max_depth + 1)
        elif action == "fixed_down":
            self.fixed_depth = max(0, self.fixed_depth - 1)
        elif action == "fixed_up":
            self.fixed_depth = min(self.max_depth, self.fixed_depth + 1)
        elif action == "fringe_down":
            self.fringe_cap = max(1, self.fringe_cap // 2)
        elif action == "fringe_up":
            self.fringe_cap = min(20_000_000, self.fringe_cap * 2)
        elif action == "time_down":
            self.time_cap_ms = max(1, self.time_cap_ms // 2)
        elif action == "time_up":
            self.time_cap_ms = min(60_000, self.time_cap_ms * 2)
        elif action == "distribution":
            self.distribution = "Fixed depth" if self.distribution == "Uniform node" else "Uniform node"
        elif action == "new_target":
            self.regenerate()
            return
        elif action == "all":
            for algorithm in ("BFS", "DFS", "IDDFS"):
                self.results[algorithm] = run_search(algorithm, self.spec, self.fringe_cap, self.time_cap_ms)
            self.message = "All algorithms used the same generated target."
            return
        elif action in ("BFS", "DFS", "IDDFS"):
            self.results[action] = run_search(action, self.spec, self.fringe_cap, self.time_cap_ms)
            self.message = f"{action} completed."
            return
        self.update_spec()
        self.message = "Parameters changed; the target was regenerated from the same seed."

    def draw_text(self, text, x, y, font=None, color=TEXT):
        self.screen.blit((font or self.font).render(text, True, color), (x, y))

    def draw(self):
        self.screen.fill(BG)
        self.draw_text("Tree Search: BFS vs DFS vs IDDFS", 26, 24, self.bold)
        self.draw_text("The tree is procedural: nodes are (depth, base-b path code), never stored as a full tree.", 26, 52, self.small, MUTED)
        pygame.draw.rect(self.screen, PANEL, (18, 92, 490, 495), border_radius=8)
        pygame.draw.rect(self.screen, PANEL, (526, 92, 496, 495), border_radius=8)

        labels = [
            f"Branching factor: {self.branching_factor}",
            f"Maximum depth: {self.max_depth}",
            f"Fixed target depth: {self.fixed_depth}",
            f"Max fringe entries: {self.fringe_cap:,}",
            f"Max wall-clock time: {self.time_cap_ms:,} ms",
        ]
        for index, label in enumerate(labels):
            self.draw_text(label, 26, 115 + index * 43, self.small, MUTED)

        for button in self.buttons:
            if button.action == "distribution":
                button.label = f"Distribution: {'Uniform' if self.distribution == 'Uniform node' else 'Fixed'}"
            button.draw(self.screen, self.small)

        self.draw_text("Generated target", 546, 112, self.bold)
        self.draw_text(f"Mode: {self.distribution}", 546, 145, self.font)
        self.draw_text(f"Goal depth: {self.spec.goal_depth}", 546, 173, self.font)
        self.draw_text(f"Seed: {self.spec.seed}", 546, 201, self.small, MUTED)
        self.draw_text("Goal code is intentionally hidden during runs.", 546, 229, self.small, MUTED)

        headers = [(546, "Algorithm"), (674, "Status"), (790, "Expanded"), (892, "Max fringe")]
        for x, label in headers:
            self.draw_text(label, x, 291, self.small, MUTED)
        for row, algorithm in enumerate(("BFS", "DFS", "IDDFS")):
            y = 325 + row * 64
            result = self.results.get(algorithm)
            self.draw_text(algorithm, 546, y, self.bold)
            if result is None:
                self.draw_text("not run", 674, y, self.font, MUTED)
                continue
            color = GOOD if result.status == "found" else WARN if result.status == "timed out" else BAD
            self.draw_text(result.status, 674, y, self.font, color)
            self.draw_text(f"{result.expanded:,}", 790, y, self.font)
            self.draw_text(f"{result.max_fringe:,}", 892, y, self.font)
            self.draw_text(f"Wall-clock: {result.elapsed_ms:.3f} ms", 674, y + 25, self.small, MUTED)

        self.draw_text(self.message, 26, 612, self.small, MUTED)
        self.draw_text("A timeout means the wall-clock cap was reached; out of memory means the fringe cap was reached.", 26, 637, self.small, MUTED)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for button in self.buttons:
                        if button.rect.collidepoint(event.pos):
                            self.handle_action(button.action)
                            break
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


if __name__ == "__main__":
    TreeSearchApp().run()
