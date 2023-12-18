# -*- coding: utf-8 -*-
#import pcl
import numpy as np
import cv2
from PIL import Image
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title
from pylab import *
import matplotlib.pyplot as plt
import open3d as o3d
import quaternion
from scipy.spatial.transform import Rotation


def load_point_cloud(pcd_file_path):
    # 从PCD文件中读取点云数据
    point_cloud = o3d.io.read_point_cloud(pcd_file_path)
    return point_cloud
x=[]
y=[]
distance=[]    
#存放需要投影点转换成二维前的雷达坐标的x坐标（距离信息），以此为依据对投影点进行染色。
distance_3d=[]    #存放转换前雷达坐标的x坐标（距离信息）。

#file_path = "./1625124364185411489.pcd"
file_path = "./022167/1625126565904426991.pcd"
#file_path = "./010400/1625125389151311935.pcd"
#file_path = "./1.pcd"
cloud = load_point_cloud(file_path)
#print(len(cloud.points))
#im = Image.open('./000150/stereo/000150.png')
im = Image.open('./022167/022167.png')
#im = Image.open('./010300/stereo/010300.png')
#im = Image.open('./010400/010400.png')
pix = im.load()
#print('pix',im.size)
points_3d = []
#print(len(cloud.points))
for i in range(len(cloud.points)):
    x_raw = cloud.points[i][0]
    y_raw = cloud.points[i][1]
    z_raw = cloud.points[i][2]
    point_3d = []

    point_3d.append(x_raw)
    point_3d.append(y_raw)
    point_3d.append(z_raw)
    
   # if y_raw > 0:
    points_3d.append(point_3d)
    distance_3d.append(y_raw)
cube = np.float64(points_3d)
#print('cube的形状',cube.shape)
#print(cube)
# 外参信息
lidar_quaternion = [0.99826517514320456, -0.051288851007357569, 0.028412105004075816, -0.005370861000770469]
lidar_translation = [5.2480345660000003, -0.064328983000000006, -0.38866764500000001]

stereo_quaternion = [0.50451874772890937, 0.49271166335600353, 0.50320304941532457, 0.4994824732081517]
stereo_translation = [5.0362549728344614, -0.44101021275849878, -1.771000242852925]

camera_matrix = np.float64([[1814.5061420730431, 0, 1026.1977130691339],
                            [0, 1814.5061420730431, 524.7984412358129],
                            [0, 0, 1]])  #内参矩阵

distCoeffs = np.float64([-0.082292808267338341,
         0.1036602731851739,
         -0.0034940032414059268,
         0.00084014002308057311,
         0.093700291321559645])  #畸变参数



# 构建激光雷达的旋转矩阵和平移向量
lidar_rotation = Rotation.from_quat(lidar_quaternion)
lidar_rotation_matrix = lidar_rotation.as_matrix()
lidar_translation_vector = np.array(lidar_translation)

#激光雷达到ahrs
lidar_to_ahrs_transform = np.eye(4)
lidar_to_ahrs_transform[:3,:3] = lidar_rotation_matrix
lidar_to_ahrs_transform[:3,3] = lidar_translation_vector

# 构建相机的旋转矩阵和平移向量
stereo_rotation = Rotation.from_quat(stereo_quaternion)
stereo_rotation_matrix = stereo_rotation.as_matrix()
stereo_translation_vector = np.array(stereo_translation)

#相机到ahrs
stereo_to_ahrs_transform = np.eye(4)
stereo_to_ahrs_transform[:3,:3] = stereo_rotation_matrix
stereo_to_ahrs_transform[:3,3] = stereo_translation_vector

ahrs_to_stereo_transform = np.linalg.inv(stereo_to_ahrs_transform)

lidar_to_stereo_transform =np.dot(ahrs_to_stereo_transform, lidar_to_ahrs_transform)

lidar_to_stereo_rotation_matrix = lidar_to_stereo_transform[:3, :3]
#print(lidar_to_stereo_rotation_matrix)
lidar_to_stereo_rotation_verctor = cv2.Rodrigues(lidar_to_stereo_rotation_matrix)[0]
#平移向量
lidar_to_stereo_translation_vector = lidar_to_stereo_transform[:3, 3]

rvec = np.float64(lidar_to_stereo_rotation_verctor)
tvec = np.float64(lidar_to_stereo_translation_vector)



point_2d, _ = cv2.projectPoints(cube, rvec, tvec, camera_matrix, distCoeffs)
# print(point_2d.shape)
# print(rvec)
# print(tvec)
#print(point_2d[0].shape)
point_2d_sque = np.squeeze(point_2d)

m=-1
for point in point_2d:
    m=m+1
    #print(m)
    x_2d = point[0][0]
    y_2d = point[0][1]

    if 0<=x_2d<2048 and 0<=y_2d<1080:
        x.append(x_2d)
        y.append(y_2d)
        distance.append(-distance_3d[m]*100)#数值取反是为了让colormap颜色由红到蓝显示而非由蓝到红
        RGB=pix[x_2d,y_2d]


x=np.array(x)
y=np.array(y)
plt.scatter(x, y, c=distance, cmap='jet',s=1,marker='.')
plt.imshow(im)
plt.show()



