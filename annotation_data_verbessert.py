import argparse
import os
import yaml
import bagpy
import ply as ply
import pandas as pd
import numpy as np
import json
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
from glob import glob
import json 
from itertools import groupby

idx2obj = {
             1: 'COW',
             2: 'GIRA',
             3: 'ORANGE',
             }
def obj2idx(obj_name): 
    for k, v in idx2obj.items():
        if v == obj_name:
            return k
def load_data(camera_path, object_path):
    camera_recording = load_rosbag(camera_path)
    object_poses = load_rosbag(object_path)
    return camera_recording, object_poses


def load_rosbag(path):
    path = os.path.abspath(path)
    bag = bagpy.bagreader(path)
    return bag


def load_pointclouds(path):
    pointclouds= ply.read_ply(path)
    return pointclouds


def load_camera_intrinsics(path):
    with open(path) as f:
        camera_intrinsics = yaml.load(f, Loader=yaml.FullLoader)
    camera_intrinsics["intrinsics"] = np.array(camera_intrinsics['camera_matrix']['data']).reshape(3, 3)
    camera_intrinsics["focal_length"] = [camera_intrinsics["intrinsics"][0, 0], camera_intrinsics["intrinsics"][1, 1]]
    camera_intrinsics["principal_point"] = [camera_intrinsics["intrinsics"][0, 2], camera_intrinsics["intrinsics"][1, 2]]
    camera_intrinsics["image_size"] = [camera_intrinsics["image_height"], camera_intrinsics["image_width"]]
    return camera_intrinsics


def calcualte_object_pose(object_data):
    """
    Return the average pose of a single object
    :param object_data:
    :return:
    """
    object_pose = {}
    means = object_data.mean(0)
    # object_pose["position"]["mean_x"] = means['pose.pose.position.x']
    # object_pose["position"]["mean_y"] = means['pose.pose.position.y']
    # object_pose["position"]["mean_z"] = means['pose.pose.position.z']
    object_pose["position"] = np.array([means['pose.pose.position.x'], means['pose.pose.position.y'], means['pose.pose.position.z']])
    # object_pose["orientation"]["mean_or_w"] = means['pose.pose.orientation.w']
    # object_pose["orientation"]["mean_or_x"] = means['pose.pose.orientation.x']
    # object_pose["orientation"]["mean_or_y"] = means['pose.pose.orientation.y']
    # object_pose["orientation"]["mean_or_z"] = means['pose.pose.orientation.z']
    object_pose["quaternion"] = [means['pose.pose.orientation.w'], means['pose.pose.orientation.x'],
                                 means['pose.pose.orientation.y'], means['pose.pose.orientation.z']]
    object_pose["rotation"] = quat2rot(object_pose["quaternion"])
    return object_pose


def quat2rot(q):
    """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
    # Extract the values from Q
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def annotation(object_poses,point_clouds): 
    obj=[]
    for topic in object_poses.topics:
        object_name = topic[1:topic.find('_rb')]
        

def center_objects(data,marker_offset=[0, 0, -0.027]):
    """
    TODO: write comment
    """
    # Iterate over each object and calculate the inverse transformation
    objects={}
    object_pose = calcualte_object_pose(data)
    rot = object_pose["rotation"]
    quat=object_pose["quaternion"]
    trans = np.array([object_pose["position"]])+marker_offset
    inv_transf = np.hstack((rot.T, -np.matmul(rot.T, trans.T))) ## Warum ot_(wo)=R^t_(WO)*qt_(wo)? Warum ein -
    inv_transf = np.vstack([inv_transf, [0, 0, 0, 1]])
    objects["center_t"] = inv_transf ## Was ist mit dem Marker Offset
    return rot, trans, objects["center_t"],quat


def center_camera(camera_recording, ct_o, marker_offset=[0, 0, -0.027]):
    camera_poses = pd.read_csv(camera_recording.message_by_topic('/synchronizer/camera_poses'))

    # Determine Rotation matrix
    camera_q_w = camera_poses["pose.pose.orientation.w"].to_numpy()
    camera_q_x = camera_poses["pose.pose.orientation.x"].to_numpy()
    camera_q_y = camera_poses["pose.pose.orientation.y"].to_numpy()
    camera_q_z = camera_poses["pose.pose.orientation.z"].to_numpy()
    camera_q = np.vstack([camera_q_w, camera_q_x, camera_q_y, camera_q_z]).T
    camera_rot = np.apply_along_axis(quat2rot, 1, camera_q)
    #print(camera_q)
    #Determine Position + Offset
    camera_x = camera_poses["pose.pose.position.x"].to_numpy()
    camera_y = camera_poses["pose.pose.position.y"].to_numpy()
    camera_z = camera_poses["pose.pose.position.z"].to_numpy()
    camera_position = np.vstack([camera_x, camera_y, camera_z]).T
    camera_trans = camera_position + marker_offset

    # Create camera transformation
    camera_tf = []
    for rot, trans in zip(camera_rot, camera_trans):
        tf = np.hstack((rot, np.array([trans]).T))
        tf = np.vstack([tf, [0, 0, 0, 1]])
        camera_tf.append(tf)
    camera_tf = np.array(camera_tf)

    # Iterate over all objects and calculate the object specific centered camera pose

    camera_object_poses = np.matmul(ct_o, camera_tf)
    C_O= camera_object_poses
    return C_O, camera_rot, camera_q,camera_trans


def calculate_relative_transformation( camera_q,object_q):
    q_relative_to_cam1=np.array([[camera_rot[0],camera_rot[0],camera_rot[2],camera_rot[3]]])
    q_relative_to_cam2=np.array([[-camera_rot[1],camera_rot[0],camera_rot[3],-camera_rot[2]]])
    q_relative_to_cam3=np.array([[-camera_rot[2],-camera_rot[3],camera_rot[0],camera_rot[1]]])
    q_relative_to_cam4=np.array([[-camera_rot[3],camera_rot[2],-camera_rot[1],camera_rot[0]]])
    q_cw=np.hstack(q_relative_to_cam1,q_relative_to_cam2,q_relative_to_cam3,q_relative_to_cam4)
    q_co=np.matmul(q_cw, object_q)
    #relative_translation_relative_to_word=
    return q_co


def create_extrinsic_matrix(camera_pose):
    if camera_pose.ndim > 2:
        camera_pose = camera_pose[0]
    camera_rot = camera_pose[0:3, 0:3].T
    extrinsics = np.hstack((camera_rot, -np.matmul(camera_rot, np.array([camera_pose[0:3, 3]]).T)))
    return extrinsics


def bounding_box(x_coord, y_coord):
    return np.rint(np.array([[np.min(x_coord), np.min(y_coord)], [np.max(x_coord), np.max(y_coord)]]))


def calc_mask(pix_u, pix_v, image_size, shrink_factor):
    # TODO: Rework and include shrink factor
    # Make sure that point are in range of the image
    pix_u = np.array(pix_u, dtype=int)
    pix_v = np.array(pix_v, dtype=int)
    for i, u in enumerate(pix_u):
        if u < 0:
            pix_u[i] = 0
        elif u >= image_size[0]:
            pix_u[i] = image_size[0] - 1
    for i, v in enumerate(pix_v):
        if v < 0:
            pix_v[i] = 0
        elif v >= image_size[1]:
            pix_v[i] = image_size[1] - 1
    bbox = bounding_box(pix_u, pix_v)
    # Check if in range of image size
    mask = np.zeros(image_size)
    points = np.vstack([pix_u, pix_v]).T
    mask[points[:, 0], points[:, 1]] = 1
    return bbox, mask


def create_mask(camera_intrinsics, pc_pos,C_O, shrink_factor=0.6):
    objects={}
    r_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    r_z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    intrinsics_matrix = camera_intrinsics["intrinsics"]
    r = np.matmul(r_z, r_x)
    constant_tf = np.matmul(intrinsics_matrix, r)
    extrinsics = create_extrinsic_matrix(C_O)
    extrinsics_tf = np.matmul(constant_tf, extrinsics)
    p = np.matmul(extrinsics_tf, pc_pos)
    u_i = p[0, :]/p[2, :]
    v_i = p[1, :]/p[2, :]
    bbox, mask = calc_mask(u_i, v_i, camera_intrinsics["image_size"], shrink_factor)
    objects["pix_u"] = u_i
    objects["pix_v"] = v_i
    #objects["binary_masks"] = mask
    mask=binary_mask_to_rle(mask)
    objects['mask']=mask
    objects["bboxs"] = bbox

    return objects

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle


def create_coco_style_data(images,plc_pths,length1,args):
    images_dic=[]
    annotation=[]
    models=[]
    C_stack=np.empty((4,4))
    length=len(images)
    camera_recording, object_poses = load_data(args.camera_path, args.object_path)
    t=0
    for plc_path in plc_pths: 
        pos=plc_path.find('_pc_')
        #models["instance_id"]=obj2idx(plc_path[length1["plc_pths"]+1:pos])
        pxx=load_pointclouds(plc_path)['points']['x'].to_numpy()
        pyy=load_pointclouds(plc_path)['points']['y'].to_numpy()
        pzz=load_pointclouds(plc_path)['points']['z'].to_numpy()
        px=pxx.tolist()
        py=pyy.tolist()
        pz=pzz.tolist()
        p=np.vstack([pxx, pyy, pzz, np.ones(len(pxx))])
        topic='/'+plc_path[length1["plc_pths"]+1:pos]+'_rb/vrpn_client/estimated_odometry'
        data = pd.read_csv(object_poses.message_by_topic(topic))
        rotation_o, translation_o, co_w,quat=center_objects(data)
        model_rot=rotation_o.tolist()
        model_pos=translation_o.tolist()
        #print(C_stack.shape)
        #print([[model_pose]])
        #dic0={'instance_id': obj2idx(plc_path[length1["plc_pths"]+1:pos]),'quaternions': [model_rot], 'postition':[model_pos],
        #      'px': [px], 'py':[py],'pz':[pz]}
        dic0={'instance_id': obj2idx(plc_path[length1["plc_pths"]+1:pos]),'quaternions_w_o': [quat], "Rotation_w_o":[model_rot],'postition_w_o':[model_pos], 
            "Trafo_o_w":[co_w.tolist()]}
        models.append(dic0)
        C_O,camera_rotation, camera_q,camera_trans = center_camera(camera_recording, co_w)
       

        j=0
        
        for name in images[:1]:
            camera_intrinsics= load_camera_intrinsics(args.camera_yaml)
            rel_posi=translation_o-camera_trans[j:j+1]
            rel_posi=[rel_posi.tolist()]
            var=length+int(name[-10:-4])
            output_mask=create_mask(camera_intrinsics, p,C_O[j:j+1], shrink_factor=0.6)
            dic1={"file_name":"%06i.png" % var, "Image_id":var, "width":camera_intrinsics["image_width"], "height":camera_intrinsics["image_height"],'camera_intrinsics': [camera_intrinsics["intrinsics"].tolist()]}
            #dict3={"Cam_R":[camera_rotation[j:j+3,:,:].tolist()],'C_O':[C_O[j:j+4].tolist()],'instance_id': obj2idx(plc_path[length1["plc_pths"]+1:pos]),
            #"id":var,"pixel u":[output_mask['pix_u'].tolist()],"pixel v":[output_mask['pix_v'].tolist()],"binary_masks":[output_mask['binary_masks'].tolist()],"bboxs":[output_mask['bboxs'].tolist()]}
            dict3={"image_id":var, 'instance_id': obj2idx(plc_path[length1["plc_pths"]+1:pos]),"R_w_c":[camera_rotation[j:j+1,:,:].tolist()],'RT_C_two_O':[C_O[j:j+1].tolist()],"q_w_c": [camera_q[j:j+1].tolist()],"position_c_w":[camera_trans[j:j+1].tolist()],"t_c_o":rel_posi,
             "mask":output_mask['mask']['counts'],"bbox":[output_mask['bboxs'].tolist()]}
            if t%len(plc_pths)==0:
                #print(t)
                images_dic.append(dic1)
            t=t+1
            annotation.append(dict3)
            j=j+1

        data={'images':images_dic,'Object_annotation':models,'annotation':annotation}
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)
    #jason["info"]["annotations"].append({})
    return


def save_images_from_bag(camera_path,image_topic,path):
    bag = rosbag.Bag(camera_path, "r")
    bridge = CvBridge()
    count = 0
    if not os.path.isdir(os.path.join(path,'rgb')):  
        os.mkdir(os.path.join(path,'rgb'))
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(os.path.join(args.output_dir, "%06i.png" % count), cv_img)

        count += 1

    bag.close()
    return


def process_rosbag(args):
    camera_recording, pt_clouds, = load_data(args.camera_path, args.object_path)
    
    
    #objects = center_camera(camera_recording, objects)
    #objects = generate_binary_masks(camera_intrinsics, objects)
    path=os.path.dirname(os.path.abspath(__file__))
    save_images_from_bag(args.camera_path,args.image_topic,path)
    rgb_pths = glob(os.path.join(path,'rgb', '*.png'))
    plc_pths=glob(os.path.join(args.pc_path,'*.ply'))
    length={}
    length["plc_pths"]=len(args.pc_path)
    length["images"]=len(rgb_pths)
    create_coco_style_data(rgb_pths,plc_pths,length,args)
    return

if __name__ == "__main__":
    path=os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_path", type=str,default=os.path.join(path,'MUKISANO_recordings/2d_frames/ANIMALS_2dscann_undframes256x256_SYNC.bag'),  help="Path to the rosbag to be processed")
    parser.add_argument("--pc_path", type=str, help="Path to the directory containing point clouds", default=os.path.join(path,'MUKISANO_recordings/PCs_filtered'))
    parser.add_argument("--object_path", type=str, help="Path to the rosbag containing the objects' poses",default=os.path.join(path,'MUKISANO_recordings/2d_frames/ANIMALS_2dscan_pose.bag') )
    parser.add_argument("--camera_yaml", type=str, help="Path to the yaml-file containing the camera intrinsics",default='calib_file_256x256.yaml')
    parser.add_argument("--output_dir",default=os.path.join(path,'rgb'), help="Output directory for recorded images.")
    parser.add_argument("--image_topic",default="/synchronizer/rgb_undistort",help="Image topic.")
    args = parser.parse_args()
    process_rosbag(args)


