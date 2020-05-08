import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from dotmap import DotMap

from humanav.humanav_renderer import HumANavRenderer
from humanav.renderer_params import create_params as create_base_params
from humanav.renderer_params import get_surreal_texture_dir

from utils.visualization import render_rgb_and_depth

class DataGenerator(object):
    def __init__(self):
        self.p = self.create_params()
        self._r = HumANavRenderer.get_renderer(self.p)
        self.initialize_occupancy_map_for_grid()
        self.rng = np.random.RandomState(69)

    def create_params(self, area_name='area5a', width=1024, height=1024, fov_v=75., fov_h=75., cam_elevation_deg=-10):
        p = create_base_params()

        # Set any custom parameters
        p.building_name = area_name

        p.camera_params.width = width
        p.camera_params.height = height
        p.camera_params.fov_vertical = fov_v
        p.camera_params.fov_horizontal = fov_h

        # The camera is assumed to be mounted on a robot at fixed height
        # and fixed pitch. See humanav/renderer_params.py for more information

        # Tilt the camera 10 degree down from the horizontal axis
        p.robot_params.camera_elevation_degree = -10

        p.camera_params.modalities = ['rgb', 'disparity']
        p.map_origin = [0,0]
        return p

    def initialize_occupancy_map_for_grid(self):
        """
        Initialize the occupancy grid for the entire map and
        associated parameters/ instance variables
        """
        resolution, traversible = self._r.get_config()

        self.p.dx = resolution / 100.  # To convert to metres.

        # Reverse the shape as indexing into the traverible is reversed ([y, x] indexing)
        self.p.map_size = np.array(traversible.shape[::-1])

        # [[min_x, min_y], [max_x, max_y]]
        self.map_bounds = np.array([[0., 0.],  self.p.map_size*self.p.dx])

        free_xy = np.array(np.where(traversible)).T
        self.free_xy_map= free_xy[:, ::-1]

        self.occupancy_grid_map = np.logical_not(traversible)*1.
    
    def sample_point(self,rng, free_xy_map=None):
        """
        Samples a real world x, y point in free space on the map.
        Optionally the user can pass in free_xy_m2 a list of m (x, y)
        points from which to sample.
        """
        if free_xy_map is None:
            free_xy_map = self.free_xy_map

        idx = rng.choice(len(free_xy_map))
        pos = free_xy_map[idx][None, None]
        return self._map_to_point(pos)

    def _map_to_point(self, pos, dtype=np.float32):
        """
        Convert pos_2 in map coordinates
        to a real world coordinate.
        """
        world_pos = (pos + self.p.map_origin)*self.p.dx
        return world_pos.astype(dtype)

    def sample_free_position(self):
        """
        samples a pose (x,y,theta) on free area in traversible
        """
        free_pos = np.empty(shape=(1,3))
        x_y = self.sample_point(self.rng)
        #theta = 2*np.pi*np.random.rand()
        theta = np.pi
        free_pos[0,:2] = x_y
        free_pos[0, 2] = theta
        return free_pos

    def plot_random_scene(self):
        identity_1 = np.random.RandomState(np.random.randint(100))
        identity_2 = np.random.RandomState(np.random.randint(100))
        start_camera_pos = self.sample_free_position()
        start_human_pos_1 = (start_camera_pos + np.array([[-2,0.5,-np.pi]]))[0]
        start_human_pos_2 = (start_camera_pos + np.array([[-2,-0.5,-np.pi]]))[0]
        h_speed = 0.7
        self._r.add_human_at_position_with_speed(start_human_pos_1, h_speed, identity_1, self.rng)
        self._r.add_human_at_position_with_speed(start_human_pos_2, h_speed, identity_2, self.rng)
        self._r.remove_human()
        rgb_image_1mk3, _ = render_rgb_and_depth(self._r, start_camera_pos, self.p.dx, human_visible=True)
        rgb_img = rgb_image_1mk3[0].astype(np.uint8)
        plt.imshow(rgb_img)
        plt.show()


    
if __name__ == '__main__':
    data_gen = DataGenerator()
    data_gen.plot_random_scene()
    print('bitches in the club')


