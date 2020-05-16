import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from dotmap import DotMap
from humanav.humanav_renderer import HumANavRenderer
from humanav.renderer_params import create_params as create_base_params
from humanav.renderer_params import get_surreal_texture_dir

def plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m,
                camera_pos_13, human_poses, filename):

    # Compute the real_world extent (in meters) of the traversible
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent)*dx_m

    fig = plt.figure(figsize=(30, 10))

    # Plot the 5x5 meter occupancy grid centered around the camera
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(traversible, extent=extent, cmap='gray',
              vmin=-.5, vmax=1.5, origin='lower')

    # Plot the camera
    ax.plot(camera_pos_13[0, 0], camera_pos_13[0, 1], 'bo', markersize=10)
    ax.quiver(camera_pos_13[0, 0], camera_pos_13[0, 1], np.cos(camera_pos_13[0, 2]), np.sin(camera_pos_13[0, 2]))
    # Plot the human
    i = 1
    for human_pos in human_poses:
        ax.plot(human_pos[0], human_pos[1], 'ro', markersize=10)
        ax.quiver(human_pos[0], human_pos[1], np.cos(human_pos[2]), np.sin(human_pos[2]))
        i+=1 

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
    f=667.2513908946974
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


def plot_traversible(traversible, dx_m, camera_pos, human_poses, goal_pos):
    plt.figure(figsize=(20,10))
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent)*dx_m
    plt.imshow(traversible, extent=extent, cmap='gray', origin='lower')

    # Plot the camera
    plt.plot(camera_pos[0, 0], camera_pos[0, 1], 'bX', markersize=5)
    #plt.quiver(camera_pos[0, 0], camera_pos[0, 1], np.cos(camera_pos[0, 2]), np.sin(camera_pos[0, 2]), color='blue')
    # Plot the human
    for human_pos in human_poses:
        plt.plot(human_pos[0], human_pos[1], 'ro', markersize=5)
        #plt.quiver(human_pos[0], human_pos[1], np.cos(human_pos[2]), np.sin(human_pos[2]), color='red')

    # Plot Goal
    plt.plot(goal_pos[0], goal_pos[1], 'gX', markersize=5)
    plt.title('Topview')
    plt.show()

