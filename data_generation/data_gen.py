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
        #self._r.add_human_at_position_with_speed(start_human_pos_2, h_speed, identity_2, self.rng)
        #self._r.remove_human()
        rgb_image_1mk3, _ = render_rgb_and_depth(self._r, start_camera_pos, self.p.dx, human_visible=True)
        rgb_img = rgb_image_1mk3[0].astype(np.uint8)
        plt.imshow(rgb_img)
        plt.show()


    
if __name__ == '__main__':
    data_gen = DataGenerator()
    data_gen.plot_random_scene()
    print('bitches in the club')


""" Commented Sequence Generation for Rex """
"""
def generate_movement_sequence(area, c_pos, h_pos, h_speed, c_speed, identity, pose, total_time, num_images, out_dir, seq_name):
    
    -Code to load a sequence of images for a moving object.
    -For now we assume that the camera is fixed and the person is moving in a straight line.
    -Also assume fixed starting position for camera and human
    -Only works for area3
    
    Inputs:
    - area (string): name of the map we are working with
    - h_speed (float): speed of the human (m/s)
    - c_speed (float): speed of the camera (m/s)
    - identity (int): seed for selecting human identity
    - pose (int): seed for selecting human pose
    - total_time (float): total time in which we collect video data (sec)
    - num_images (int): number of images to capture over total time interval 
    - out_dir (string): directory for saving image sequence folder
    - seq_name (string): name of images in sequence (will be enumerated)


    Outputs:
    should save a collection of images


    Goals:
    - Get this working for a moving camera
    - Make this work with non-fixed starting positions for robot and human
    - Make this work with different areas besides area3
    - Make this work with multiple humans
    - Make human animate as they walk (just need to change meshes)

    Long-term:
    - Make this work with an arbitrary trajectory (need to adjust path planning)
    
    # Setup
    assert os.path.isdir(out_dir), "Error: Provided directory {} for saving image sequence does not exist.".format(out_dir)
    folder_path = os.path.join(out_dir, seq_name)
    rgb_path = os.path.join(folder_path, 'rgb')
    depth_path = os.path.join(folder_path, 'depth')
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if not os.path.isdir(rgb_path):
        os.mkdir(rgb_path)
    if not os.path.isdir(depth_path):
        os.mkdir(depth_path)   
    
    p = create_params()
    
    r = HumANavRenderer.get_renderer(p)
    dx_cm, traversible = r.get_config()
    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    # Set the identity seed. This is used to sample a random human identity
    # (gender, texture, body shape)
    identity_rng = np.random.RandomState(identity)

    # Set the Mesh seed. This is used to sample the actual mesh to be loaded
    # which reflects the pose of the human skeleton.
    mesh_rng = np.random.RandomState(pose)

    # Specify starting state for camera and human.
    # Specified as [x (meters), y (meters), theta (radians)] coordinates
    start_camera_pos = np.array([[7.5, 12., -1.3]])
    start_human_pos = np.array([8.0, 9.75, np.pi/2])

    delta_t = total_time/num_images
    human_pos = start_human_pos
    cam_pos = start_camera_pos
   
    r.add_human_at_position_with_speed(start_human_pos, h_speed, identity_rng, mesh_rng)
    for i in range(num_images):

        human_pos[0] = i*delta_t*h_speed*np.cos(start_human_pos[2]) + start_human_pos[0]
        human_pos[1] = i*delta_t*h_speed*np.sin(start_human_pos[2]) + start_human_pos[1]

        cam_pos[0,0] = i*delta_t*c_speed*np.cos(start_camera_pos[0,2]) + start_camera_pos[0,0]
        cam_pos[0,1] = i*delta_t*c_speed*np.sin(start_camera_pos[0,2]) + start_camera_pos[0,1]
        
        #r.add_human_with_known_identity_at_position_with_speed(start_human_pos, human_speed, mesh_rng, human_identity, True)
        r.move_human_to_position_with_speed(human_pos, h_speed, mesh_rng)
        rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, cam_pos, dx_m, human_visible=True)
        print(depth_image_1mk1.shape)
        depth_img = (depth_image_1mk1[0, :, :, 0]).astype(np.uint16)
        
        rgb_img = cv2.cvtColor(rgb_image_1mk3[0].astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Save image
        #plt.imshow(rgb_img)
        #plt.show()
        rgb_file = os.path.join(rgb_path, 'img{}.png'.format(i+1))
        depth_file = os.path.join(depth_path, 'img{}.png'.format(i+1))
        cv2.imwrite(rgb_file, rgb_img)
        cv2.imwrite(depth_file, depth_img)
        

    # Remove the human from the environment
    print('success!')





if __name__ == '__main__':
    #test1()
    #example1() 
    #example2() 
    
    generate_movement_sequence(area='area3', h_speed=1, c_speed=0.4, identity=48, 
                                pose=20, total_time=1, num_images=2, 
                                out_dir='/home/shawnshact/ucb/ee106b/seq', seq_name='test_moving')