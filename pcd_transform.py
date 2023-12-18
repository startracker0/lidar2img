import numpy as np
import cv2
from PIL import Image
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title
from pylab import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d


def visualize_point_cloud(bin_file_path,save_pcb_path):   
    lidar_dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('time', np.uint32), ('reflectivity', np.uint16), ('ambient', np.uint16), ('range', np.uint32)]
    scan = np.fromfile(bin_file_path, dtype=lidar_dtype)

    points = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)
    
    # Creating an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Save the PointCloud object to PCB format
    o3d.io.write_point_cloud(save_pcb_path, point_cloud)

    return points


visualize_point_cloud('./010400/1625125389151311935.bin','./010400/1625125389151311935.pcd')