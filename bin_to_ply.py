import os
import numpy as np
import open3d as o3d

# 입력 및 출력 경로
bin_dir = "/home/soeun/LiDAR_snow_sim/nuScenes_intensity"
ply_output_dir = "/home/soeun/LiDAR_snow_sim/output_ply"
label_output_dir = "/home/soeun/LiDAR_snow_sim/output_label"

os.makedirs(ply_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)

# 모든 .bin 파일 처리
for filename in os.listdir(bin_dir):
    if filename.endswith(".bin"):
        bin_path = os.path.join(bin_dir, filename)
        base_name = os.path.splitext(filename)[0]

        # .bin 파일 읽기
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
        xyz = points[:, :3]
        intensity = points[:, 3]
        labels = points[:, 4].astype(np.uint32)

        # ----------- PLY 파일 생성 -----------
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # intensity를 grayscale RGB로 매핑
        intensity_norm = (intensity - intensity.min()) / (intensity.ptp() + 1e-5)
        colors = np.stack([intensity_norm] * 3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        ply_path = os.path.join(ply_output_dir, f"{base_name}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)

        # ----------- LABEL 파일 생성 -----------
        label_path = os.path.join(label_output_dir, f"{base_name}.label")
        labels.tofile(label_path)

        print(f"Processed {filename} -> {base_name}.ply + {base_name}.label")
