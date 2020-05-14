import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from dotmap import DotMap
from humanav.humanav_renderer import HumANavRenderer
from humanav.renderer_params import create_params as create_base_params
from humanav.renderer_params import get_surreal_texture_dir

def create_params():
    p = create_base_params()

    # Set any custom parameters
    p.building_name = 'area3'
    p.camera_params.width = 1024
    p.camera_params.height = 1024
    p.camera_params.fov_vertical = 75.
    p.camera_params.fov_horizontal = 75.

    # The camera is assumed to be mounted on a robot at fixed height
    # and fixed pitch. See humanav/renderer_params.py for more information
    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.camera_elevation_degree = -10
    p.camera_params.modalities = ['rgb', 'disparity']
    return p
def render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=True, rgb = True, depth = True):
    # Convert from real world units to grid world units
    camera_grid_world_pos_12 = camera_pos_13[:, :2]/dx_m

    # Render RGB and Depth Images. The shape of the resulting
    # image is (1 (batch), m (width), k (height), c (number channels))
    rgb_image_1mk3, depth_image_1mk1 = None, None
    if rgb:
        rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=True)

    if depth:
        depth_image_1mk1, _, _ = r._get_depth_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], xy_resolution=.05, map_size=1500, pos_3=camera_pos_13[0, :3], human_visible=True)

    return rgb_image_1mk3, depth_image_1mk1

def get_known_human_identity(r):
    """
    Specify a known human identity. An identity
    is a dictionary with keys {'human_gender', 'human_texture', 'body_shape'}
    """

    # Method 1: Set a seed and sample a random identity
    identity_rng = np.random.RandomState(48)
    human_identity = r.load_random_human_identity(identity_rng)

    # Method 2: If you know which human you want to load,
    # specify the params manually (or load them from a file)
    human_identity = {'human_gender': 'male', 'human_texture': [os.path.join(get_surreal_texture_dir(), 'train/male/nongrey_male_0110.jpg')], 'body_shape': 1320}
    return human_identity

class Trajectory:
    def __init__(self, area, h_pos, c_pos, h_speed, c_speed, identity, pose, total_time, num_images, out_dir, seq_name):
        self.area = area
        self.h_speed = h_speed
        self.c_speed = c_speed
        self.pose = pose
        self.total_time = total_time
        self.num_images = num_images
        self.out_dir = out_dir
        self.seq_name = seq_name
        self.dt = total_time/num_images
        self.r = HumANavRenderer.get_renderer(create_params())
        self.identity = identity
        self.h_pos = h_pos
        self.c_pos = c_pos
        self.__setupDirectory()

    def __setupDirectory(self):
        # Setup
        assert os.path.isdir(
            self.out_dir), "Error: Provided directory {} for saving image sequence does not exist.".format(
            self.out_dir)
        self.folder_path = os.path.join(self.out_dir, self.seq_name)
        self.rgb_path = os.path.join(self.folder_path, 'rgb')
        self.depth_path = os.path.join(self.folder_path, 'depth')
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
        if not os.path.isdir(self.rgb_path):
            os.mkdir(self.rgb_path)
        if not os.path.isdir(self.depth_path):
            os.mkdir(self.depth_path)

    def get_linear_trajectory(self):
        p = create_params()

        r = HumANavRenderer.get_renderer(p)
        dx_cm, traversible = r.get_config()
        # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
        dx_m = dx_cm / 100.

        # Set the identity seed. This is used to sample a random human identity
        # (gender, texture, body shape)
        identity_rng = np.random.RandomState(self.identity)

        # Set the Mesh seed. This is used to sample the actual mesh to be loaded
        # which reflects the pose of the human skeleton.
        mesh_rng = np.random.RandomState(self.pose)

        # Specify starting state for camera and human.
        # Specified as [x (meters), y (meters), theta (radians)] coordinates
        start_human_pos = self.h_pos
        human_pos = self.h_pos
        cam_pos = self.c_pos
        r.add_human_at_position_with_speed(start_human_pos, self.h_speed, identity_rng, mesh_rng)

        for i in range(self.num_images):
            human_pos[0] = i * self.dt * self.h_speed * np.cos(start_human_pos[2]) + start_human_pos[0]
            human_pos[1] = i * self.dt * self.h_speed * np.sin(start_human_pos[2]) + start_human_pos[1]
            cam_pos[0, 0] = i * self.dt * self.c_speed * np.cos(start_camera_pos[0, 2]) + start_camera_pos[0, 0]
            cam_pos[0, 1] = i * self.dt * self.c_speed * np.sin(start_camera_pos[0, 2]) + start_camera_pos[0, 1]
            r.move_human_to_position_with_speed(human_pos, self.h_speed, mesh_rng)
            rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, cam_pos, dx_m, human_visible=True)
            depth_img = depth_image_1mk1[0, :, :, 0]
            rgb_img = cv2.cvtColor(rgb_image_1mk3[0].astype(np.uint8), cv2.COLOR_RGB2BGR)
            rgb_file = os.path.join(self.rgb_path, 'img{}.png'.format(i + 1))
            depth_file = os.path.join(self.depth_path, 'img{}.png'.format(i + 1))
            cv2.imwrite(rgb_file, rgb_img)
            cv2.imwrite(depth_file, depth_img)

    def get_circle_trajectory(self):
        p = create_params()

        r = HumANavRenderer.get_renderer(p)
        dx_cm, traversible = r.get_config()
        # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
        dx_m = dx_cm / 100.

        # Set the identity seed. This is used to sample a random human identity
        # (gender, texture, body shape)
        identity_rng = np.random.RandomState(self.identity)

        # Set the Mesh seed. This is used to sample the actual mesh to be loaded
        # which reflects the pose of the human skeleton.
        mesh_rng = np.random.RandomState(self.pose)

        # Specify starting state for camera and human.
        # Specified as [x (meters), y (meters), theta (radians)] coordinates
        start_human_pos = self.h_pos
        human_pos = self.h_pos
        cam_pos = self.c_pos
        r.add_human_at_position_with_speed(start_human_pos, self.h_speed, identity_rng, mesh_rng)
        circle_center = np.array([human_pos[0] - 5, human_pos[1]])
        curr_h_angle = human_pos[2]
        d_theta = np.pi/self.num_images
        for i in range (self.num_images):
            human_pos[2] = i * d_theta + start_human_pos[2]
            human_pos[0] = np.cos(human_pos[2]) + start_human_pos[0]
            human_pos[1] = np.sin(human_pos[2]) + start_human_pos[1]
            r.move_human_to_position_with_speed(human_pos, self.h_speed, mesh_rng)
            rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, cam_pos, dx_m, human_visible=True)
            depth_img = depth_image_1mk1[0, :, :, 0]
            rgb_img = cv2.cvtColor(rgb_image_1mk3[0].astype(np.uint8), cv2.COLOR_RGB2BGR)
            rgb_file = os.path.join(self.rgb_path, 'img_cir{}.png'.format(i + 1))
            depth_file = os.path.join(self.depth_path, 'img_cir{}.png'.format(i + 1))
            cv2.imwrite(rgb_file, rgb_img)
            cv2.imwrite(depth_file, depth_img)
            print("Image {}".format(i))

if __name__ == '__main__':
    start_camera_pos = np.array([[7.5, 15., -1.5]])
    start_human_pos = np.array([8.0, 9.75, np.pi/2])
    a = Trajectory(area='area3', h_pos=start_human_pos, c_pos=start_camera_pos, h_speed=1, c_speed=0,
                   identity=48, pose=20, total_time=1, num_images=10, out_dir='/home/rex/test', seq_name='test_moving')
    a.get_linear_trajectory()
    # a.get_circle_trajectory()