import gym
import numpy as np
from PIL import Image
from minigrid.wrappers import *
from mini_behavior.window import Window
from mini_behavior.utils.save import get_step, save_demo
from mini_behavior.grid import GridDimension
from mini_behavior.states import *
from collections import deque
import random

TILE_PIXELS = 32

class MiniBehaviorEnv:
    def __init__(self, env_id='MiniGrid-InstallingAPrinter-8x8-N2-v0', seed=-1, tile_size=32,
                 agent_view=False, save_demo_flag=False, load_path=None):

        self.env_id = env_id
        self.seed = seed
        self.tile_size = tile_size
        self.agent_view = agent_view
        self.save_demo_flag = save_demo_flag
        self.load_path = load_path
        self.show_furniture = False
        self.all_steps = {}

        self.env = gym.make(env_id)
        self.env.teleop_mode()
        self.key_to_action = {
            '0': self.env.actions.pickup_0,
            '1': self.env.actions.pickup_1,
            '2': self.env.actions.pickup_2,
            '3': self.env.actions.drop_0,
            '4': self.env.actions.drop_1,
            '5': self.env.actions.drop_2,
            't': self.env.actions.toggle,
            'o': self.env.actions.open,
            'c': self.env.actions.close,
            'k': self.env.actions.cook,
            '6': self.env.actions.slice,
            'i': self.env.actions.drop_in,
        }
        for obj_type, obj_list in self.env.objs.items():
            for obj in obj_list:
                self.key_to_action["moveto-" + obj.name] = "moveto-" + obj.name
        

        if self.agent_view:
            self.env = RGBImgPartialObsWrapper(self.env)
            self.env = ImgObsWrapper(self.env)

        self.window = Window('mini_behavior - ' + env_id)
        self.window.no_closeup()

        if self.load_path is not None:
            self._load_state()

        self.nav_sampler_cache = {}
        self.short_task = True

    def redraw(self, img):
        if not self.agent_view:
            img = self.env.render()
        self.window.set_inventory(self.env)
        self.window.show_img(img)
        self.window.save_img("output_image.jpeg")

    def render_furniture(self):
        self.show_furniture = not self.show_furniture
        if self.show_furniture:
            img = np.copy(self.env.furniture_view)
            i, j = self.env.agent_pos
            ymin = j * TILE_PIXELS
            ymax = (j + 1) * TILE_PIXELS
            xmin = i * TILE_PIXELS
            xmax = (i + 1) * TILE_PIXELS
            img[ymin:ymax, xmin:xmax, :] = GridDimension.render_agent(
                img[ymin:ymax, xmin:xmax, :], self.env.agent_dir)
            img = self.env.render_furniture_states(img)
            self.window.show_img(img)
        else:
            obs = self.env.gen_obs()
            self.redraw(obs)

    def show_states(self):
        imgs = self.env.render_states()
        self.window.show_closeup(imgs)

    def switch_dim(self, dim):
        self.env.switch_dim(dim)
        print(f'switching to dim: {self.env.render_dim}')
        obs = self.env.gen_obs()
        self.redraw(obs)

    def _load_state(self):
        if self.seed != -1:
            self.env.seed(self.seed)
        self.env.reset()
        obs = self.env.load_state(self.load_path)
        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)
            self.window.set_caption(self.env.mission)
        self.redraw(obs)

    def reset(self):
        if self.seed != -1:
            self.env.seed(self.seed)
        obs = self.env.reset()
        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)
            self.window.set_caption(self.env.mission)
        self.redraw(obs)
        return obs

    def get_lifted_state(self):
        objs = self.env.objs
        obj_instances = {}
        for obj_type, obj_list in objs.items():
            for obj in obj_list:
                obj_instances[obj.name] = obj

        ground_atoms = []
        for k, o in obj_instances.items():
            for pred_name, pred in o.states.items():
                if isinstance(pred, (AbsoluteObjectState, AbilityState, ObjectProperty)):
                    if pred.get_value(self.env):
                        ground_atoms.append(f"{pred_name}({k})")
                elif isinstance(pred, RelativeObjectState):
                    for k2, o2 in obj_instances.items():
                        if o.check_rel_state(self.env, o2, pred_name):
                            ground_atoms.append(f"{pred_name}({k},{k2})")
        return ground_atoms

    def step(self, action):
        prev_obs = self.env.gen_obs()
        prev_state = self.get_lifted_state()
        if isinstance(action, str) and action.startswith("moveto-"):
            self.move_in_front_of(action.replace("moveto-","")) 
            obs = self.env.gen_obs()
            reward = 0.0
            done = False
            terminated = False
            info = {}
        else:
            obs, reward, done, terminated, info = self.env.step(action)
            if self.short_task:
                if self.env_id == 'MiniGrid-SortingBooks-16x16-N2-v0':
                    book = self.env.objs['book']
                    hardback = self.env.objs['hardback']
                    shelf = self.env.objs['shelf'][0]
                    for obj in book + hardback:
                        if obj.check_rel_state(self.env, shelf, 'onTop'):
                            reward = 1.0
                            done = 1.0
                elif self.env_id == 'MiniGrid-WateringHouseplants-16x16-N2-v0':
                    pot_plants = self.env.objs['pot_plant']
                    for plant in pot_plants:
                        if plant.check_abs_state(self.env, 'soakable'):
                            reward = 1.0
                            done = 1.0
                elif self.env_id == 'MiniGrid-ThrowingAwayLeftovers-16x16-N2-v0':
                    for hamburger in self.env.objs['hamburger']:
                        is_inside = [hamburger.check_rel_state(self.env, ashcan, 'inside') for ashcan in self.env.objs['ashcan']]
                        if True in is_inside:
                            reward = 1.0
                            done = 1.0
                elif self.env_id == 'MiniGrid-BoxingBooksUpForStorage-16x16-N2-v0':
                    book = self.env.objs['book']
                    box = self.env.objs['box'][0]
                    for obj in book:
                        if obj.check_rel_state(self.env, box, 'inside'):
                            reward = 1.0
                            done = 1.0
        state = self.get_lifted_state()

        print(f'env_id={self.env_id}, step={self.env.step_count}, reward={reward:.2f}')
        # for atom in state:
        #     print(atom)

        if self.save_demo_flag:
            self.all_steps[self.env.step_count] = (prev_obs, prev_state, action, obs, state)

        if done:
            print('done!')
            if self.save_demo_flag:
                save_demo(self.all_steps, self.env_id, self.env.episode)
            self.reset()
        else:
            self.redraw(obs)
        return obs, reward, done, terminated, info

    def show(self):
        self.window.show(block=False)

    def key_handler_primitive(self, event):
        print('pressed', event.key)
        action_map = {
            'left': self.env.actions.left,
            'right': self.env.actions.right,
            'up': self.env.actions.forward,
            '0': self.env.actions.pickup_0,
            '1': self.env.actions.pickup_1,
            '2': self.env.actions.pickup_2,
            '3': self.env.actions.drop_0,
            '4': self.env.actions.drop_1,
            '5': self.env.actions.drop_2,
            't': self.env.actions.toggle,
            'o': self.env.actions.open,
            'c': self.env.actions.close,
            'k': self.env.actions.cook,
            '6': self.env.actions.slice,
            'i': self.env.actions.drop_in
        }

        if event.key == 'escape':
            self.window.close()
        elif event.key in action_map:
            self.step(action_map[event.key])
        elif event.key == 'pagedown':
            self.show_states()

    def bfs_path(self, start, goal):
        grid = self.env.grid
        width, height = grid.width, grid.height
        visited = set()
        queue = deque([(start, [])])
        
        while queue:
            current_pos, path = queue.popleft()
            if current_pos == goal:
                return path

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = current_pos[0] + dx, current_pos[1] + dy
                next_pos = (nx, ny)

                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if next_pos in visited:
                    continue
                if grid.get(nx, ny) != [[None, None], [None, None], [None, None]]:
                    if grid.get(nx, ny)[0][0] is None or grid.get(nx, ny)[0][0].name != "door":
                        continue  # Obstacle

                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
        
        return None  # No path found

    def move_in_front_of(self, target_name):
        # Find the target object
        target_obj = None
        for obj_list in self.env.objs.values():
            for obj in obj_list:
                if obj.name == target_name:
                    target_obj = obj
                    break
            if target_obj:
                break

        if target_obj is None:
            print(f"[Error] Object '{target_name}' not found.")
            return

        reachable = []
        start_pos = tuple(self.env.agent_pos)
        target_pos = target_obj.cur_pos
        adjacents = [
            (target_pos[0] + 1, target_pos[1]),
            (target_pos[0] - 1, target_pos[1]),
            (target_pos[0], target_pos[1] + 1),
            (target_pos[0], target_pos[1] - 1)
        ]
        pos_to_target = {}
        for adj in adjacents:
            pos_to_target[adj] = target_pos
        if hasattr(target_obj, 'all_pos'):
            adjacents = []
            for target_pos in target_obj.all_pos:
                if 'cabinet' not in target_name:
                    if target_name in self.nav_sampler_cache:
                        if target_pos in self.nav_sampler_cache[target_name]:
                            continue
                new_adjacents = [
                    (target_pos[0] + 1, target_pos[1]),
                    (target_pos[0] - 1, target_pos[1]),
                    (target_pos[0], target_pos[1] + 1),
                    (target_pos[0], target_pos[1] - 1)
                ]
                for adj in new_adjacents:
                    pos_to_target[adj] = target_pos
                adjacents += new_adjacents
        # Choose a reachable adjacent position
        random.shuffle(adjacents)
        for pos in adjacents:
            if (0 <= pos[0] < self.env.grid.width and 0 <= pos[1] < self.env.grid.height):
                if self.env.grid.get(*pos) == [[None, None], [None, None], [None, None]]:
                    path = self.bfs_path(start_pos, pos)
                    if path:
                        reachable.append((pos, path))
                elif self.env.grid.get(*pos)[0][0] is not None:
                    if self.env.grid.get(*pos)[0][0].name == "door":
                        path = self.bfs_path(start_pos, pos)
                        if path:
                            reachable.append((pos, path))
                else:
                    pass

        if not reachable:
            print(f"[Error] No accessible position next to '{target_name}'")
            return

        # Choose shortest reachable
        goal_pos, path = min(reachable, key=lambda x: len(x[1]))

        # Follow path
        for next_pos in path:
            dx = next_pos[0] - self.env.agent_pos[0]
            dy = next_pos[1] - self.env.agent_pos[1]

            desired_dir = {
                (1, 0): 0,
                (0, 1): 1,
                (-1, 0): 2,
                (0, -1): 3
            }.get((dx, dy))

            if desired_dir is None:
                continue

            while self.env.agent_dir != desired_dir:
                self.step(self.env.actions.right)
            self.step(self.env.actions.forward)

        # Face the object
        target_pos = pos_to_target[tuple(self.env.agent_pos)]
        face_dir = (target_pos[0] - self.env.agent_pos[0], target_pos[1] - self.env.agent_pos[1])
        target_dir = {
            (1, 0): 0,
            (0, 1): 1,
            (-1, 0): 2,
            (0, -1): 3
        }.get(face_dir)

        if target_dir is not None:
            while self.env.agent_dir != target_dir:
                self.step(self.env.actions.right)

        print(f"[Success] Reached position in front of '{target_name}', facing it.")
        if target_name in self.nav_sampler_cache:
            self.nav_sampler_cache[target_name].append(target_pos)
        else:
            self.nav_sampler_cache[target_name] = [target_pos]

