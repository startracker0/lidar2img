import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import transformations as tf
# lidar_dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('time', np.uint32), ('reflectivity', np.uint16), ('ambient', np.uint16), ('range', np.uint32)]

# scan = np.fromfile('./010300/1625125379152256637.bin', dtype=lidar_dtype)

# points = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)

# # 可视化点云
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, c='r', marker='.')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#plt.show()


def load_lidar_data(file_path):
    # 定义数据类型
    lidar_dtype = [
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
        ('time', np.uint32),
        ('reflectivity', np.uint16),
        ('ambient', np.uint16),
        ('range', np.uint32)
    ]

    try:
        # 从二进制文件中读取数据
        scan = np.fromfile(file_path, dtype=lidar_dtype)

        # 提取坐标信息 (x, y, z)
        points = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)

        return points
    except IOError as e:
        print(f"Error loading lidar data from {file_path}: {e}")
        return None


def project_lidar_to_image(lidar_data):
    # 将激光雷达数据投影到图像上

# 外参信息
    lidar_quaternion = [0.99826517514320456, -0.051288851007357569, 0.028412105004075816, -0.005370861000770469]
    lidar_translation = [5.2480345660000003, -0.064328983000000006, -0.38866764500000001]

    stereo_quaternion = [0.50451874772890937, 0.49271166335600353, 0.50320304941532457, 0.4994824732081517]
    stereo_translation = [5.0362549728344614, -0.44101021275849878, -1.771000242852925]


    
    # 内部参数
    cc_x = 1026.1977130691339
    cc_y = 524.7984412358129
    focal_length_x = 1814.5061420730431
    focal_length_y = 1814.5061420730431
    image_width = 2048
    image_height = 1080

    # 构建相机内参矩阵
    K = np.array([
        [focal_length_x, 0, cc_x],
        [0, focal_length_y, cc_y],
        [0, 0, 1]
    ])
    
# 畸变系数
    #distortion_coefficients = [-0.082292808267338341, 0.1036602731851739, -0.0034940032414059268, 0.00084014002308057311, 0.093700291321559645]
    
    # 构建激光雷达的变换矩阵
    lidar_transform_matrix = tf.quaternion_matrix(lidar_quaternion)
    lidar_transform_matrix[:3, 3] = lidar_translation

    # 构建立体相机的变换矩阵
    stereo_transform_matrix = tf.quaternion_matrix(stereo_quaternion)
    stereo_transform_matrix[:3, 3] = stereo_translation
    
    # 构建从激光雷达到相机的整体变换矩阵
    lidar_to_stereo_transform = np.dot(np.linalg.inv(stereo_transform_matrix), lidar_transform_matrix)
    
    #激光雷达齐次坐标
    lidar_data_homogeneous = np.hstack((lidar_data, np.ones((lidar_data.shape[0], 1))))

    # 将激光雷达齐次坐标转换到相机坐标系    camera_coords 现在包含了每个激光雷达点在相机坐标系中的坐标
    camera_coords = np.dot(lidar_to_stereo_transform, lidar_data_homogeneous.T).T
    
    
    # 将相机坐标系中的点转换为齐次坐标...
    camera_coords_homogeneous = np.hstack((camera_coords, np.ones((camera_coords.shape[0], 1))))

    # 使用cv2.undistortPoints处理畸变
    distorted_points = camera_coords_homogeneous[:, :2].reshape((-1, 1, 2))
    
    # 将畸变系数转换为NumPy数组
    distortion_coefficients = np.array([-0.082292808267338341, 0.1036602731851739, -0.0034940032414059268, 0.00084014002308057311, 0.093700291321559645])

    # 使用cv2.undistortPoints处理畸变
    undistorted_points = cv2.undistortPoints(distorted_points, K, distortion_coefficients, P=K)
    undistorted_points_homogeneous = np.hstack((undistorted_points.squeeze(), np.ones((undistorted_points.shape[0], 1))))
    
    
    # 将激光雷达-相机坐标映射到相机图像坐标
    image_points_homogeneous = np.dot(K, undistorted_points_homogeneous.T).T
    
    # 归一化坐标
    image_points_normalized = image_points_homogeneous[:, :2] / image_points_homogeneous[:, 2:]

    # 转换为像素坐标，并取整
    image_points_pixel = (image_points_normalized * np.array([image_width, image_height])).astype(int)

    
    return image_points_pixel    


visible_img = cv2.imread('./000050/stereo/000050.png')
points =load_lidar_data('./000050/1625124354187668691.bin')
lidar_data_image = project_lidar_to_image(points)
# 创建一个图形
# plt.figure(figsize=(10, 8))

# # 显示可见光图像
# plt.imshow(cv2.cvtColor(visible_img, cv2.COLOR_BGR2RGB))

# # 在可见光图像上绘制激光雷达数据
# plt.scatter(lidar_data_image[:, 0], lidar_data_image[:, 1], c='g', marker='.', s=1)  # 绿色的散点
# plt.imshow(visible_img, extent=plt.xlim() + plt.ylim(), aspect='auto', zorder=-1)
# plt.axis('off')  # 关闭坐标轴

# plt.show()
#plt.savefig('output_plot.png')

# 显示图像
plt.imshow(cv2.cvtColor(visible_img, cv2.COLOR_BGR2RGB))
# 获取图像大小

# 设置 scatter 函数的 alpha 为 1
plt.scatter(lidar_data_image[:, 0], lidar_data_image[:, 1], c='g', marker='.', s=1)

# 显示图像
plt.show()