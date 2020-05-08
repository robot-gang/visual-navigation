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


def plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m,
                camera_pos_13, human_pos_3, filename):

    # Compute the real_world extent (in meters) of the traversible
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent)*dx_m

    fig = plt.figure(figsize=(30, 10))

    # Plot the 5x5 meter occupancy grid centered around the camera
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(traversible, extent=extent, cmap='gray',
              vmin=-.5, vmax=1.5, origin='lower')

    # Plot the camera
    ax.plot(camera_pos_13[0, 0], camera_pos_13[0, 1], 'bo', markersize=10, label='Camera')
    ax.quiver(camera_pos_13[0, 0], camera_pos_13[0, 1], np.cos(camera_pos_13[0, 2]), np.sin(camera_pos_13[0, 2]))
    # Plot the human
    ax.plot(human_pos_3[0], human_pos_3[1], 'ro', markersize=10, label='Human')
    ax.quiver(human_pos_3[0], human_pos_3[1], np.cos(human_pos_3[2]), np.sin(human_pos_3[2]))

    ax.legend()
    ax.set_xlim([camera_pos_13[0, 0]-5., camera_pos_13[0, 0]+5.])
    ax.set_ylim([camera_pos_13[0, 1]-5., camera_pos_13[0, 1]+5.])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Topview')

    # Plot the RGB Image
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('RGB')

    # Plot the Depth Image
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(depth_image_1mk1[0, :, :, 0].astype(np.uint8), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Depth')

    fig.savefig(filename, bbox_inches='tight', pad_inches=0)


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


def example1():
    """
    Code for loading a random human into the environment
    and rendering topview, rgb, and depth images.
    """
    p = create_params()
    
    r = HumANavRenderer.get_renderer(p)
    dx_cm, traversible = r.get_config()
    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    # Set the identity seed. This is used to sample a random human identity
    # (gender, texture, body shape)
    identity_rng = np.random.RandomState(10)

    # Set the Mesh seed. This is used to sample the actual mesh to be loaded
    # which reflects the pose of the human skeleton.
    mesh_rng = np.random.RandomState(10)

    # State of the camera and the human. 
    # Specified as [x (meters), y (meters), theta (radians)] coordinates
    camera_pos_13 = np.array([[7.5, 12., -1.3]])
    human_pos_3 = np.array([8.0, 9.75, np.pi])

    # Speed of the human in m/s
    human_speed = 0.7

    # Load a random human at a specified state and speed
    r.add_human_at_position_with_speed(human_pos_3, human_speed, identity_rng, mesh_rng)

    # Get information about which mesh was loaded
    human_mesh_info = r.human_mesh_params

    rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=True)

    # Remove the human from the environment
    r.remove_human()

    # Plot the rendered images
    plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m,
                camera_pos_13, human_pos_3, 'example1.png')
    print('success!')


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

def example2():
    """
    Code for loading a specified human identity into the environment
    and rendering topview, rgb, and depth images.
    Note: Example 2 is expected to produce the same output as Example1
    """
    p = create_params()

    r = HumANavRenderer.get_renderer(p)
    dx_cm, traversible = r.get_config()

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    human_identity = get_known_human_identity(r)

    # Set the Mesh seed. This is used to sample the actual mesh to be loaded
    # which reflects the pose of the human skeleton.
    mesh_rng = np.random.RandomState(20)

    # State of the camera and the human. 
    # Specified as [x (meters), y (meters), theta (radians)] coordinates
    camera_pos_13 = np.array([[7.5, 12., -1.3]])
    human_pos_3 = np.array([8.0, 9.75, np.pi/2.])

    # Speed of the human in m/s
    human_speed = 0.7

    # Load a random human at a specified state and speed
    r.add_human_with_known_identity_at_position_with_speed(human_pos_3, human_speed, mesh_rng, human_identity)

    # Get information about which mesh was loaded
    #human_mesh_info = r.human_mesh_params

    rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=True)
    # Remove the human from the environment
    r.remove_human()

    # Plot the rendered images
    plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m,
                camera_pos_13, human_pos_3, 'example2.png')
    print('success')


def generate_movement_sequence(area, h_speed, c_speed, identity, pose, total_time, num_images, out_dir, seq_name):
    """
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
    """
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
                                pose=20, total_time=1, num_images=20, 
                                out_dir='/home/rex/test', seq_name='test_moving')
