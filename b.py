import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from tqdm import tqdm


# 读取地图配置文件
def read_map_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        lines = f.readlines()
        map_info = {}
        for line in lines:
            key, value = line.strip().split(": ")
            if key == "image":
                map_info[key] = value
            elif key == "resolution":
                map_info[key] = float(value)
            elif key == "origin":
                map_info[key] = [float(x) for x in value.strip("[]").split(", ")]
    return map_info


# 处理激光雷达扫描数据
def process_scan(scan_msg, map_info):
    scan_points = []
    angle = scan_msg["angle_min"]
    for r in scan_msg["ranges"]:
        if scan_msg["range_min"] <= r <= scan_msg["range_max"]:
            x = r * math.cos(angle) / map_info["resolution"]
            y = -r * math.sin(angle) / map_info["resolution"]
            scan_points.append((x, y))
        angle += scan_msg["angle_increment"]
    return scan_points


# 计算匹配值
def calculate_matching_value(scan_points, lidar_x, lidar_y, lidar_yaw, map_temp):
    transform_points = []
    clockwise_points = []
    counter_points = []
    deg_to_rad = math.pi / 180.0

    for point in scan_points:
        # 情况一：原始角度
        rotated_x = point[0] * math.cos(lidar_yaw) - point[1] * math.sin(lidar_yaw)
        rotated_y = point[0] * math.sin(lidar_yaw) + point[1] * math.cos(lidar_yaw)
        transform_points.append((rotated_x + lidar_x, lidar_y - rotated_y))

        # 情况二：顺时针旋转1度
        clockwise_yaw = lidar_yaw + deg_to_rad
        rotated_x = point[0] * math.cos(clockwise_yaw) - point[1] * math.sin(
            clockwise_yaw
        )
        rotated_y = point[0] * math.sin(clockwise_yaw) + point[1] * math.cos(
            clockwise_yaw
        )
        clockwise_points.append((rotated_x + lidar_x, lidar_y - rotated_y))

        # 情况三：逆时针旋转1度
        counter_yaw = lidar_yaw - deg_to_rad
        rotated_x = point[0] * math.cos(counter_yaw) - point[1] * math.sin(counter_yaw)
        rotated_y = point[0] * math.sin(counter_yaw) + point[1] * math.cos(counter_yaw)
        counter_points.append((rotated_x + lidar_x, lidar_y - rotated_y))

    offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    point_sets = [transform_points, clockwise_points, counter_points]
    yaw_offsets = [0, deg_to_rad, -deg_to_rad]

    max_sum = 0
    best_dx = 0
    best_dy = 0
    best_dyaw = 0

    for i in range(len(offsets)):
        for j in range(len(point_sets)):
            sum_val = 0
            for point in point_sets[j]:
                px = int(point[0] + offsets[i][0])
                py = int(point[1] + offsets[i][1])
                if 0 <= px < map_temp.shape[1] and 0 <= py < map_temp.shape[0]:
                    sum_val += map_temp[py, px]
            if sum_val > max_sum:
                max_sum = sum_val
                best_dx = offsets[i][0]
                best_dy = offsets[i][1]
                best_dyaw = yaw_offsets[j]

    return best_dx, best_dy, best_dyaw


# 主函数
def main():
    scan_data_path = "./data/sim/raw_data.npz"
    map_info = read_map_yaml("./data/sim/map.yaml")
    scan_data = np.load(scan_data_path)["scan"]
    map_raw = cv2.imread(map_info["image"], cv2.IMREAD_GRAYSCALE)
    lidar_x = 0
    lidar_y = 0
    lidar_yaw = 0
    scan_pose = []
    for scan in tqdm(scan_data[:1000]):
        scan_msg = {
            "angle_min": -math.pi,
            "angle_max": math.pi,
            "angle_increment": math.pi / 320,
            "range_min": 0.1,
            "range_max": 12.0,
            "ranges": scan,
        }
        scan_points = process_scan(scan_msg, map_info)
        while True:
            best_dx, best_dy, best_dyaw = calculate_matching_value(
                scan_points, lidar_x, lidar_y, lidar_yaw, map_raw
            )
            lidar_x += best_dx
            lidar_y += best_dy
            lidar_yaw += best_dyaw
            if abs(best_dx) < 1 and abs(best_dy) < 1 and abs(best_dyaw) < 0.01:
                break
        scan_pose.append((lidar_x, lidar_y))

    scan_pose = np.array(scan_pose)
    plt.plot(scan_pose[:, 0], scan_pose[:, 1], marker="o")
    plt.title("Lidar Scan Trajectory")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
