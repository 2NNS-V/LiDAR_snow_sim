import numpy as np
import open3d as o3d

bin_path = '/home/soeun/LiDAR_snow_sim/simulated_snowflake_output.bin'
points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)

points.astype(np.float32).tofile('/home/soeun/LiDAR_snow_sim/nuScenes_soeun/nuScenes_intensity/001401.bin')