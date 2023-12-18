import numpy as np
lidar_dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('time', np.uint32), ('reflectivity', np.uint16), ('ambient', np.uint16), ('range', np.uint32)]

scan = np.fromfile("./022167/1625126565904426991.bin", dtype=lidar_dtype)

points = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)
print(points.shape)