import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
from dotmap import DotMap

from humanav.humanav_renderer import HumANavRenderer
from humanav.renderer_params import create_params as create_base_params
from humanav.renderer_params import get_surreal_texture_dir

from utils.visualization import render_rgb_and_depth, plot_images, plot_traversible
class Human(object):
    def __init__(self, speed = None, identity_seed = None, mesh_seed = None, id=None):
        if id is None:
            self.id = uuid.uuid4().hex # generate completely random id
        else:
            self.id = id
        
        if speed is None:
            self.speed = 1.7*np.random.rand()
        else:
            self.speed = speed

        if identity_seed is None:
            self.identity_rng = np.random.RandomState(np.random.randint(100))
        else:
            self.identity_rng = np.random.RandomState(identity_seed)
        
        if mesh_seed is None:
            self.mesh_rng = np.random.RandomState(np.random.randint(100))
        else:
            self.mesh = np.random.RandomState(mesh_seed)
        

class DataGenerator(object):
    def __init__(self, cam_angle=None, cam_rng=None, h_rng=None):
        self.p = self.create_params()
        self._r = HumANavRenderer.get_renderer(self.p)
        self.initialize_occupancy_map_for_grid()
        
        if cam_angle is None:
            self.cam_angle = 2*np.pi*np.random.rand()
        else:
            self.cam_angle = cam_angle % (2*np.pi)

        if cam_rng is None:
            self.cam_rng = np.random.RandomState(np.random.randint(100))
        else:
            self.cam_rng = np.random.RandomState(cam_rng)
        
        if h_rng is None:
            self.h_rng = np.random.RandomState(np.random.randint(100))
        else:
            self.h_rng = np.random.RandomState(h_rng)

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
        self.traversible = traversible
        self.p.dx = resolution / 100.  # To convert to metres.

        # Reverse the shape as indexing into the traverible is reversed ([y, x] indexing)
        self.p.map_size = np.array(traversible.shape[::-1])

        # [[min_x, min_y], [max_x, max_y]]
        self.map_bounds = np.array([[0., 0.],  self.p.map_size*self.p.dx])

        free_xy = np.array(np.where(traversible)).T
        self.free_xy_map= free_xy[:, ::-1]

        self.occupancy_grid_map = np.logical_not(traversible)*1.
    
    def sample_point(self, rng, free_xy_map=None):
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

    def sample_free_position(self, rng, mode='camera'):
        """
        samples a pose (x,y,theta) on free area in traversible
        """
        free_pos = np.empty(shape=(1,3))
        x_y = self.sample_point(rng)
        if mode.lower() != 'camera' and mode.lower() != 'cam':
            theta = 2*np.pi*np.random.rand()
        else:
            theta = self.cam_angle
        free_pos[0,:2] = x_y
        free_pos[0, 2] = theta
        if mode.lower() != 'camera' and mode.lower() != 'cam':
            return free_pos[0] 
        return free_pos

    def plot_random_scene(self, num_humans=50):
        startpos_h_lst = [self.sample_free_position(self.h_rng, mode='human') for i in range(num_humans)]
        humans = [Human() for i in range(num_humans)]
        for i in range(num_humans):
            self._r.add_human_at_position_with_speed(humans[i], startpos_h_lst[i])

        startpos_cam = self.sample_free_position(self.cam_rng, mode='cam') 
        goal_pos = self.sample_free_position(self.cam_rng, mode='goal')
        rgb_image_1mk3, depth = render_rgb_and_depth(self._r, startpos_cam, self.p.dx, human_visible=True, depth=True)
        #rgb_img = rgb_image_1mk3[0].astype(np.uint8)
        plot_name = 'test{}.png'.format(uuid.uuid4().hex)
        plot_images(rgb_image_1mk3, depth, self._r.building.traversible, self.p.dx, startpos_cam, startpos_h_lst, plot_name)
        #plot_traversible(self._r.building.map.traversible, self.p.dx, startpos_cam, startpos_h_lst, goal_pos)


if __name__ == '__main__':
    data_gen = DataGenerator(cam_angle=np.pi, cam_rng=69)
    human1, human2, human3 = [Human() for i in range(3)]
    startpos_cam = data_gen.sample_free_position(data_gen.cam_rng, mode='cam')
    h_pos1 = startpos_cam[0]+[-2,-0.5,np.pi]
    h_pos2 = startpos_cam[0]+[-3,0,np.pi]
    h_pos3 = startpos_cam[0]+[-2,0.5,np.pi]
    h_lst = [h_pos1, h_pos2, h_pos3]
    data_gen._r.add_human_at_position_with_speed(human1, h_pos1)
    data_gen._r.add_human_at_position_with_speed(human2, h_pos2)
    data_gen._r.add_human_at_position_with_speed(human3, h_pos3)
    rgb_image_1mk3, depth = render_rgb_and_depth(data_gen._r, startpos_cam, data_gen.p.dx, human_visible=True, depth=True)
    plot_name = 'test{}.png'.format(uuid.uuid4().hex)
    plot_images(rgb_image_1mk3, depth, data_gen._r.building.traversible, data_gen.p.dx, startpos_cam, h_lst, plot_name)


# properly update traversal maps