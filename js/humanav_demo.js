var remote_base_url = 'https://humanav.ddns.net';

// Variables to track current "state" of the renderer
var current_topview_plot_center = [7.5, 12.];
// var current_identity_seed = 48;
var current_mesh_seed = 20;
var current_robot_pos_3 = [7.5, 12., -1.3];
var current_human_pos_3 = [8.0, 9.75, Math.PI/2.0];
var current_human_speed = .7;
var current_human_identity = {'human_gender': 'male',
                              'body_shape': 1320,
                              'human_texture': ['/home/ext_drive/somilb/data/surreal/code/surreal/datageneration/smpl_data/visual_mpc_textures/train/male/nongrey_male_0110.jpg']}

var current_img_urls = {'human_visible': {
                                    'rgb': './resources/images/humanav_demo/human_visible/default_rgb.png',
                                    'topview': './resources/images/humanav_demo/human_visible/default_topview.png',
                                    'depth': './resources/images/humanav_demo/human_visible/default_depth.png'
                                },
                        'human_not_visible': {
                                    'rgb': './resources/images/humanav_demo/human_not_visible/default_rgb.png',
                                    'topview': './resources/images/humanav_demo/human_not_visible/default_topview.png',
                                    'depth': './resources/images/humanav_demo/human_not_visible/default_depth.png'
                          },
                        'topview_unoccupied': "./resources/images/humanav_demo/default_topview_unoccupied.png"};

// Variables for the topview animation
var half_width = 5.0; // Topview plot is 10 x 10 meters
var horizontal_center = 112.5;
var horizontal_half_width = 125.5;
var vertical_center = 105;
var vertical_half_width = 126;
var theta_offset = -1.3;

var control_panel_closed = true;

function changeHumanSpeed(){
    /*
        Update the current speed. Assumes the integer valued bound are
        [0, 60] which map to [0.0, .60] m/s. This is clamped interally
        on the server, so no point in trying to spoof it :).
    */

    current_human_speed = parseFloat(document.getElementById('human_speed').value)/100.0;
    document.getElementById('current_human_speed_text').innerHTML = "(" + current_human_speed.toFixed(2) + " m/s)";
}

function changeRobotXState(){
    var rel_robot_x_coor = parseFloat(document.getElementById('robot_relative_x_coor').value)/100.0;
    var new_robot_icon_x_coor = horizontal_center - rel_robot_x_coor*horizontal_half_width;
    var robot_icon = document.getElementById('topview_img_robot_icon');
    robot_icon.style.right = new_robot_icon_x_coor.toFixed(1) + "px";

    var robot_x_coor = current_topview_plot_center[0] + rel_robot_x_coor*half_width;
    current_robot_pos_3[0] = robot_x_coor;
}

function changeRobotYState(){
    var rel_robot_y_coor = parseFloat(document.getElementById('robot_relative_y_coor').value)/100.0;
    var new_robot_icon_y_coor = vertical_center + rel_robot_y_coor*vertical_half_width;
    var robot_icon = document.getElementById('topview_img_robot_icon');
    robot_icon.style.top = new_robot_icon_y_coor.toFixed(1) + "px";

    var robot_y_coor = current_topview_plot_center[1] - rel_robot_y_coor*half_width;
    current_robot_pos_3[1] = robot_y_coor;
}

function changeRobotThetaState(){
    var rel_robot_theta_coor = parseFloat(document.getElementById('robot_relative_theta_coor').value)/100.0;
    var new_robot_icon_theta_coor_radians = angle_normalize(rel_robot_theta_coor*Math.PI + theta_offset);
    var robot_icon = document.getElementById('topview_img_robot_icon');
    robot_icon.style.transform = "rotate(" + new_robot_icon_theta_coor_radians.toFixed(4) + "rad)";


    var robot_theta_coor = angle_normalize(rel_robot_theta_coor*Math.PI);
    current_robot_pos_3[2] = -robot_theta_coor;
}

function changeHumanXState(){
    var rel_human_x_coor = parseFloat(document.getElementById('human_relative_x_coor').value)/100.0;
    var new_human_icon_x_coor = horizontal_center - rel_human_x_coor*horizontal_half_width;
    var human_icon = document.getElementById('topview_img_human_icon');
    human_icon.style.right = new_human_icon_x_coor.toFixed(1) + "px";

    var human_x_coor = current_topview_plot_center[0] + rel_human_x_coor*half_width;
    current_human_pos_3[0] = human_x_coor;
}

function changeHumanYState(){
    var rel_human_y_coor = parseFloat(document.getElementById('human_relative_y_coor').value)/100.0;
    var new_human_icon_y_coor = vertical_center + rel_human_y_coor*vertical_half_width;
    var human_icon = document.getElementById('topview_img_human_icon');
    human_icon.style.top = new_human_icon_y_coor.toFixed(1) + "px";

    var human_y_coor = current_topview_plot_center[1] - rel_human_y_coor*half_width;
    current_human_pos_3[1] = human_y_coor;
}

function changeHumanThetaState(){
    var rel_human_theta_coor = parseFloat(document.getElementById('human_relative_theta_coor').value)/100.0;
    var new_human_icon_theta_coor_radians = angle_normalize(rel_human_theta_coor*Math.PI + theta_offset);
    var human_icon = document.getElementById('topview_img_human_icon');
    human_icon.style.transform = "rotate(" + new_human_icon_theta_coor_radians.toFixed(4) + "rad)";


    var human_theta_coor = angle_normalize(rel_human_theta_coor*Math.PI);
    current_human_pos_3[2] = -human_theta_coor;
}

function renderImages(){
    /*
        Render new images based on the settings in the
        control panel.
    */

    // Change the human identity if needed
    // var human_identity_needs_to_be_changed = document.getElementById('change_human_identity').checked;
    // if (human_identity_needs_to_be_changed) {
    //     current_identity_seed = seedUINT32();
    // }

    // Set a seed for changing various parts of the identity
    var identity_seed = seedUINT32();

    // Figure out which parts of the identity need to be changed
    var change_gender = document.getElementById('change_human_gender').checked
    var change_texture = document.getElementById('change_human_texture').checked;
    var change_body_shape = document.getElementById('change_human_body_shape').checked;

    // Change the current mesh seed if needed
    var human_pose_needs_to_be_changed = document.getElementById('change_human_appearance').checked;
    if (human_pose_needs_to_be_changed) {
        current_mesh_seed = seedUINT32();
    }
    

    // Assemble the json data needed for rendering images
    var json_data = {'robot_pos_3': current_robot_pos_3,
                     'human_pos_3': current_human_pos_3,
                     'human_speed': current_human_speed,
                     'human_identity': current_human_identity,
                     'change_human_gender': change_gender,
                     'change_human_texture': change_texture,
                     'change_body_shape': change_body_shape,
                     'identity_seed': identity_seed,
                     'mesh_seed': current_mesh_seed};
    json_data = JSON.stringify(json_data);

    // Send a request to the server to render images corresponding to these parameters
    var url = remote_base_url + '/render_images';
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            // Parse the JSON response
            data_json = JSON.parse(JSON.parse(this.response));

            // Update the human identity
            updateHumanIdentity(data_json);

            // Update the Image URL's
            setCurrImgUrls(data_json);

            // Update the current plot_center
            current_topview_plot_center[0] =  current_robot_pos_3[0];
            current_topview_plot_center[1] =  current_robot_pos_3[1];

            // Update the current images on the page
            updateCurrentImages();

            // Reset the topview animation
            resetTopViewAnimation();
        } else if (this.readyState == 4 && this.status == 400)
        {
            alert(this.responseText);
        }
    }
    xhr.open("POST", url, true);
    xhr.send(json_data);
}

function formatServerImgUrl(img_url){
    /* 
        Append the server name
        to the relative img path.
    */
    server_img_url = remote_base_url + '/' + img_url;
    return server_img_url;
}

function resetTopViewAnimation(){
    // Place the robot icon at the center of the new plot
    var robot_icon = document.getElementById('topview_img_robot_icon');
    robot_icon.style.right = horizontal_center.toFixed(1) + "px";
    robot_icon.style.top = vertical_center.toFixed(1) + "px";

    // Place the human in their new relative position
    var human_icon = document.getElementById('topview_img_human_icon');
    var rel_human_x_offset = (current_human_pos_3[0] - current_topview_plot_center[0])/half_width;
    var rel_human_y_offset = (current_human_pos_3[1] - current_topview_plot_center[1])/half_width;

    var horizontal_pixel_offset = horizontal_center - rel_human_x_offset*horizontal_half_width;
    human_icon.style.right = horizontal_pixel_offset.toFixed(1) + "px";

    var vertical_pixel_offset = vertical_center - rel_human_y_offset*vertical_half_width;
    human_icon.style.top = vertical_pixel_offset.toFixed(1) + "px";
}

function updateCurrentImages(){
    /*
        Update the current displayed images based on the data in current_img_urls
    */
    if (humanCurrentlyVisible()){
        updateImageUrls(current_img_urls['human_visible']['topview'], 
                        current_img_urls['human_visible']['rgb'],
                        current_img_urls['human_visible']['depth'],
                        current_img_urls['topview_unoccupied']);
    } else {
        updateImageUrls(current_img_urls['human_not_visible']['topview'],
                        current_img_urls['human_not_visible']['rgb'], 
                        current_img_urls['human_not_visible']['depth'],
                        current_img_urls['topview_unoccupied']);
    }

}

function updateHumanIdentity(data_json){
    current_human_identity['human_gender'] = data_json['human_identity']['human_gender'];
    current_human_identity['body_shape'] = data_json['human_identity']['body_shape'];
    current_human_identity['human_texture'] = data_json['human_identity']['human_texture'];

    // Update the options to change the human's identity to all be unchecked
    document.getElementById('change_human_gender').checked = false;
    document.getElementById('change_human_texture').checked = false;
    document.getElementById('change_human_body_shape').checked = false;
    document.getElementById('change_human_appearance').checked = false;
}

function setCurrImgUrls(img_urls){
    /*
        Set the current global variable current_img_urls
        to point to the images for the current rendering of the environment
        and human.
    */
    human_visible_urls = {'rgb': formatServerImgUrl(img_urls['human_visible']['rgb']),
                          'depth': formatServerImgUrl(img_urls['human_visible']['depth']),
                          'topview': formatServerImgUrl(img_urls['human_visible']['topview'])};
    human_not_visible_urls = {'rgb': formatServerImgUrl(img_urls['human_not_visible']['rgb']),
                            'depth': formatServerImgUrl(img_urls['human_not_visible']['depth']),
                            'topview': formatServerImgUrl(img_urls['human_not_visible']['topview'])};
    current_img_urls['human_visible'] = human_visible_urls;
    current_img_urls['human_not_visible'] = human_not_visible_urls;
    current_img_urls['topview_unoccupied'] = formatServerImgUrl(img_urls['topview_unoccupied']);
}

function humanCurrentlyVisible(){
    return document.getElementById('change_human_visibility').checked;
}

function updateImageUrls(topview_url, rgb_url, depth_url, topview_unoccupied_url){
    document.getElementById('humanav_topview_img').src = topview_url;
    document.getElementById('humanav_rgb_img').src = rgb_url;
    document.getElementById('humanav_depth_img').src = depth_url;
    document.getElementById('humanav_topview_unoccupied_img').src = topview_unoccupied_url;
}

function changeHumanVisibility(){
    updateCurrentImages();
}

function seedUINT32(){
    /*
        Create a random integer valued
        seed in the range [0, 2^32-1]
    */
    max_int = Math.pow(2, 32) - 1;
    return Math.floor(Math.random() * max_int);
}

function angle_normalize(theta){
    // Wrap an angle to the range [-pi, pi]
    return (((theta + Math.PI) % (2 * Math.PI)) - Math.PI);

}

function expandControlPanel(){
    if (control_panel_closed){
        document.getElementsByClassName("collapsible_content")[0].style.display = 'block';
        document.getElementsByClassName("live_demo_control_panel")[0].style.width = '65%';
        document.getElementsByClassName("live_demo_control_panel")[0].style.paddingBottom = '4.5%';

        // document.getElementById('control_panel_icon').innerHTML = '-';
        control_panel_closed = false;
    } else {
        document.getElementsByClassName("collapsible_content")[0].style.display = 'none';
        document.getElementsByClassName("live_demo_control_panel")[0].style.width = '20%';
        document.getElementsByClassName("live_demo_control_panel")[0].style.paddingBottom = '2%';

        // document.getElementById('control_panel_icon').innerHTML = '+';
        control_panel_closed = true;
    }


}



