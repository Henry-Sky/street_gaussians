import sqlite3
import os
import numpy as np
import glob
import cv2
import shutil
import sys
sys.path.append(os.getcwd())
import json
from scipy.spatial.transform import Rotation as R
from lib.config import cfg
from lib.utils.waymo_utils import load_camera_info
from lib.utils.data_utils import get_val_frames

image_filename_to_cam = lambda x: int(x.split('/')[0].split('_')[1])  # cam_{cam_id}/{frame}.png


def convert_filename(filename):
    # {frame}_{cam_id}.png -> cam_{cam_id}/{frame}.png
    frame, cam_id = filename.split('.')[0].split('_')
    new_filename = f'cam_{cam_id}/{frame}.png'
    return new_filename


def run_colmap_waymo(result):
    model_path = cfg.model_path
    data_path = cfg.source_path
    colmap_dir = os.path.join(model_path, 'colmap')
    os.makedirs(colmap_dir, exist_ok=True)
    print('running colmap, colmap dir: ', colmap_dir)

    unique_cams = sorted(list(set(result['cams'])))
    print('cameras: ', unique_cams)

    for unqiue_cam in unique_cams:
        train_images_dir = os.path.join(colmap_dir, 'train_imgs', f'cam_{unqiue_cam}')
        test_images_dir = os.path.join(colmap_dir, 'test_imgs', f'cam_{unqiue_cam}')
        mask_images_dir = os.path.join(colmap_dir, 'mask', f'cam_{unqiue_cam}')
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(test_images_dir, exist_ok=True)
        os.makedirs(mask_images_dir, exist_ok=True)

    train_images_dir = os.path.join(colmap_dir, 'train_imgs')
    test_images_dir = os.path.join(colmap_dir, 'test_imgs')
    mask_images_dir = os.path.join(colmap_dir, 'mask')

    image_filenames = result['image_filenames']
    c2ws = result['c2ws']
    ixts = result['ixts']
    frames_idx = result['frames_idx']
    cams = result['cams']

    split_test = cfg.data.get('split_test', -1)
    split_train = cfg.data.get('split_train', -1)
    num_frames = len(image_filenames)

    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    c2w_dict = dict()
    train_image_filenames = []
    test_image_filenames = []
    mask_image_filenames = []

    for i, image_filename in enumerate(image_filenames):
        frame_idx = frames_idx[i]
        basename = os.path.basename(image_filename)
        new_image_filename = convert_filename(basename)
        c2w_dict[new_image_filename] = c2ws[i]
        mask_image_filenames.append(os.path.join(data_path, 'dynamic_mask', basename))
        if frame_idx in train_frames:
            train_image_filenames.append(image_filename)
        if frame_idx in test_frames:
            test_image_filenames.append(image_filename)

    # copy train images
    for i, image_filename in enumerate(train_image_filenames):
        basename = os.path.basename(image_filename)
        new_image_filename = os.path.join(train_images_dir, convert_filename(basename))
        if not os.path.exists(new_image_filename):
            shutil.copyfile(image_filename, new_image_filename)

    # copy test images
    for i, image_filename in enumerate(test_image_filenames):
        basename = os.path.basename(image_filename)
        new_image_filename = os.path.join(test_images_dir, convert_filename(basename))
        if not os.path.exists(new_image_filename):
            shutil.copyfile(image_filename, new_image_filename)

    # copy mask
    for i, image_filename in enumerate(mask_image_filenames):
        basename = os.path.basename(image_filename)
        new_image_filename = os.path.join(mask_images_dir, convert_filename(basename))
        new_mask_filename = f'{new_image_filename}.png'
        if not os.path.exists(new_mask_filename):
            shutil.copyfile(image_filename, new_mask_filename)
            mask = cv2.imread(new_mask_filename)
            flip_mask = (255 - mask).astype(np.uint8)
            cv2.imwrite(new_mask_filename, flip_mask)

    print("==== run feature_extractor ====")
    ret = os.system(f'colmap feature_extractor \
            --ImageReader.mask_path {mask_images_dir} \
            --ImageReader.camera_model SIMPLE_PINHOLE  \
            --ImageReader.single_camera_per_folder 1 \
            --database_path {colmap_dir}/database.db \
            --image_path {train_images_dir}')
    if ret != 0:
        raise RuntimeError(f"feature_extractor failed, ret={ret}")

    # load intrinsic
    camera_infos = dict()
    for unique_cam in unique_cams:
        for i, cam in enumerate(cams):
            if cam == unique_cam:
                break
        sample_img = cv2.imread(image_filenames[i])
        img_h, img_w = sample_img.shape[:2]
        camera_infos[unique_cam] = {
            'ixt': ixts[i],
            'img_h': img_h,
            'img_w': img_w,
        }

    db = f'{colmap_dir}/database.db'
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('SELECT image_id, name, camera_id FROM images')
    db_images = c.fetchall()

    out_fn = f'{colmap_dir}/id_names.txt'
    with open(out_fn, 'w') as f:
        for row in db_images:
            f.write(str(row[0]) + ' ' + row[1] + '\n')

    id_names = []
    cam_to_id = dict()  # waymo_cam -> colmap_db_camera_id
    for image_id, name, db_cam_id in db_images:
        id_names.append([image_id, name, db_cam_id])
        waymo_cam = image_filename_to_cam(name)
        cam_to_id[waymo_cam] = db_cam_id
    conn.close()

    model_dir = f'{colmap_dir}/created/sparse/model'
    os.makedirs(model_dir, exist_ok=True)

    # create images.txt, use real db_cam_id
    f_w = open(f'{model_dir}/images.txt', 'w')
    for image_id, name, db_cam_id in id_names:
        print(f"[DEBUG] lookup c2w_dict key: {name}")
        if name not in c2w_dict:
            print(f"ERROR: key {name} not in c2w_dict! available keys sample: {list(c2w_dict.keys())[:8]}")
            raise KeyError(f"c2w_dict missing image key: {name}")
        transform = c2w_dict[name]
        transform = np.linalg.inv(transform)  # cam2world -> world2cam
        r = R.from_matrix(transform[:3, :3])
        rquat = r.as_quat()  # x,y,z,w scalar‑last
        rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
        out = np.concatenate((rquat, transform[:3, 3]), axis=0)
        f_w.write(f'{image_id} ')
        f_w.write(' '.join([str(a) for a in out.tolist()]))
        f_w.write(f' {db_cam_id} {name}')
        f_w.write('\n\n')
    f_w.close()

    # create cameras.txt, use real db_cam_id
    cameras_fn = os.path.join(model_dir, 'cameras.txt')
    with open(cameras_fn, 'w') as f:
        for waymo_cam in unique_cams:
            db_cam_id = cam_to_id[waymo_cam]
            camera_info = camera_infos[waymo_cam]
            ixt = camera_info['ixt']
            img_w = camera_info['img_w']
            img_h = camera_info['img_h']
            fx = ixt[0, 0]
            cx = ixt[0, 2]
            cy = ixt[1, 2]
            f.write(f'{db_cam_id} SIMPLE_PINHOLE {img_w} {img_h} {fx} {cx} {cy}\n')

    # update database camera params
    conn = sqlite3.connect(db)
    c = conn.cursor()
    for waymo_cam in unique_cams:
        cam_id = cam_to_id[waymo_cam]
        ixt = camera_infos[waymo_cam]['ixt']
        fx, cx, cy = ixt[0, 0], ixt[0, 2], ixt[1, 2]
        params = np.array([fx, cx, cy]).astype(np.float64)
        c.execute("UPDATE cameras SET params = ? WHERE camera_id = ?", (params.tostring(), cam_id))
    conn.commit()
    conn.close()

    # empty points3D.txt
    points3D_fn = os.path.join(model_dir, 'points3D.txt')
    os.system(f'touch {points3D_fn}')

    # rig config
    cam_rigid = dict()
    ref_camera_id = unique_cams[0]
    cam_rigid["ref_camera_id"] = ref_camera_id
    rigid_cam_list = []
    _, extrinsics, _, _ = load_camera_info(cfg.source_path)
    for cam_id in unique_cams:
        rigid_cam = dict()
        rigid_cam["camera_id"] = cam_id
        ref_extrinsic = extrinsics[ref_camera_id]
        cur_extrinsic = extrinsics[cam_id]
        rel_extrinsic = np.linalg.inv(cur_extrinsic) @ ref_extrinsic
        r = R.from_matrix(rel_extrinsic[:3, :3])
        qvec = r.as_quat()
        rigid_cam["image_prefix"] = 'cam_{}'.format(cam_id)
        rigid_cam['cam_from_rig_rotation'] = [qvec[3], qvec[0], qvec[1], qvec[2]]
        rigid_cam['cam_from_rig_translation'] = [rel_extrinsic[0, 3], rel_extrinsic[1, 3], rel_extrinsic[2, 3]]
        rigid_cam_list.append(rigid_cam)
    cam_rigid["cameras"] = rigid_cam_list
    rigid_config_path = os.path.join(colmap_dir, "cam_rigid_config.json")
    with open(rigid_config_path, "w+") as f:
        json.dump([cam_rigid], f, indent=4)

    print("==== run sequential_matcher ====")
    # 3.9.1 sequential_matcher不支持命令行传window_size/overlap，只传数据库路径
    ret = os.system(f'colmap sequential_matcher --database_path {colmap_dir}/database.db')
    if ret != 0:
        raise RuntimeError(f"sequential_matcher failed ret={ret}")

    triangulated_dir = os.path.join(colmap_dir, 'triangulated/sparse/model')
    os.makedirs(triangulated_dir, exist_ok=True)

    print("==== run mapper(replace point_triangulator, fix input poses) ====")
    ret = os.system(f'colmap mapper \
        --database_path {colmap_dir}/database.db \
        --image_path {train_images_dir} \
        --input_path {model_dir} \
        --output_path {triangulated_dir} \
        --Mapper.fix_existing_images 1 \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0 \
        --Mapper.filter_max_reproj_error 6 \
        --Mapper.filter_min_tri_angle 0.2')
    if ret != 0:
        raise RuntimeError(f"colmap mapper FAILED! return code: {ret}")

    if cfg.data.use_colmap_pose:
        print("==== run rig_bundle_adjuster ====")
        ret = os.system(f'colmap rig_bundle_adjuster \
                --input_path {triangulated_dir} \
                --output_path {triangulated_dir} \
                --rig_config_path {rigid_config_path} \
                --estimate_rig_relative_poses 0 \
                --RigBundleAdjustment.refine_relative_poses 1 \
                --BundleAdjustment.max_num_iterations 50 \
                --BundleAdjustment.refine_focal_length 0 \
                --BundleAdjustment.refine_principal_point 0 \
                --BundleAdjustment.refine_extra_params 0')
        if ret != 0:
            raise RuntimeError(f"rig_bundle_adjuster failed ret={ret}")

    # ⚠️Important: DO NOT delete images during colmap pipeline
    # os.system(f'rm -rf {train_images_dir}')
    # os.system(f'rm -rf {test_images_dir}')
    # os.system(f'rm -rf {mask_images_dir}')

    print("colmap pipeline finished OK.")


if __name__ == '__main__':
    run_colmap_waymo(result=None)
