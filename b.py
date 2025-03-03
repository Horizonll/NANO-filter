import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from tqdm import tqdm
from sensor_msgs.msg import LaserScan
import rclpy
from rclpy.node import Node
import time
from collections import deque

data_queue = deque(maxlen=10)  # 保持最近10个位置数据
MAX_DISTANCE = 30  # 最大允许距离变化
MAX_YAW = 5 * (math.pi / 180)  # 最大允许角度变化（5度）


def check_position(x, y, yaw):
    # 如果队列为空，直接返回False
    if not data_queue:
        data_queue.append((x, y, yaw))
        return False

    # 添加新数据
    data_queue.append((x, y, yaw))

    # 如果队列未满，继续匹配
    if len(data_queue) < data_queue.maxlen:
        return False

    # 获取第一个和最后一个数据
    first = data_queue[0]
    last = data_queue[-1]

    # 计算差值
    dx = abs(last[0] - first[0])
    dy = abs(last[1] - first[1])
    dyaw = abs(last[2] - first[2])

    # 如果所有差值都小于阈值，清空队列并返回True
    if dx < MAX_DISTANCE and dy < MAX_DISTANCE and dyaw < MAX_YAW:
        data_queue.clear()
        return True

    return False


def wait_for_message(node, topic_type, topic):
    class _vfm(object):
        def __init__(self) -> None:
            self.msg = None

        def cb(self, msg):
            self.msg = msg

    vfm = _vfm()
    subscription = node.create_subscription(topic_type, topic, vfm.cb, 1)
    while rclpy.ok():
        if vfm.msg is not None:
            return vfm.msg
        rclpy.spin_once(node)
        time.sleep(0.001)
    subscription.destroy()


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


def process_scan(scan_msg, map_info):
    angles = np.arange(
        scan_msg["angle_min"], scan_msg["angle_max"], scan_msg["angle_increment"]
    )
    ranges = np.array(scan_msg["ranges"])
    valid_indices = (ranges >= scan_msg["range_min"]) & (
        ranges <= scan_msg["range_max"]
    )
    ranges = ranges[valid_indices]
    angles = angles[valid_indices]
    x = ranges * np.cos(angles) / map_info["resolution"]
    y = -ranges * np.sin(angles) / map_info["resolution"]
    return np.column_stack((x, y))


def calculate_matching_value(scan_points, lidar_x, lidar_y, lidar_yaw, map_temp):
    deg_to_rad = math.pi / 180.0 * 2
    yaw_offsets = [0, deg_to_rad, -deg_to_rad]
    offsets = [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5)]

    max_sum = 0
    best_dx, best_dy, best_dyaw = 0, 0, 0

    for yaw_offset in yaw_offsets:
        cos_yaw = np.cos(lidar_yaw + yaw_offset)
        sin_yaw = np.sin(lidar_yaw + yaw_offset)
        rotated_points = np.dot(
            scan_points, np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        )
        transformed_points = rotated_points + np.array([lidar_x, lidar_y])

        for dx, dy in offsets:
            shifted_points = transformed_points + np.array([dx, dy])
            valid_indices = (
                (shifted_points[:, 0] >= 0)
                & (shifted_points[:, 0] < map_temp.shape[1])
                & (shifted_points[:, 1] >= 0)
                & (shifted_points[:, 1] < map_temp.shape[0])
            )
            valid_points = shifted_points[valid_indices].astype(int)
            sum_val = np.sum(map_temp[valid_points[:, 1], valid_points[:, 0]])

            if sum_val > max_sum:
                max_sum = sum_val
                best_dx, best_dy, best_dyaw = dx, dy, yaw_offset

    return best_dx, best_dy, best_dyaw


def create_gradient_mask(size):
    center = size // 2
    y, x = np.ogrid[:size, :size]
    distance = np.hypot(x - center, y - center)
    mask = np.clip(255 * (1 - distance / center), 0, 255).astype(np.uint8)
    return mask


def process_map(map_raw):
    map_temp = np.zeros_like(map_raw, dtype=np.uint8)
    gradient_mask = create_gradient_mask(91)
    mask_center = gradient_mask.shape[0] // 2

    # 获取所有为0的点的坐标
    zero_points = np.argwhere(map_raw == 0)

    for y, x in zero_points:
        left = max(0, x - mask_center)
        top = max(0, y - mask_center)
        right = min(map_raw.shape[1] - 1, x + mask_center)
        bottom = min(map_raw.shape[0] - 1, y + mask_center)
        roi = map_temp[top : bottom + 1, left : right + 1]
        mask_left = mask_center - (x - left)
        mask_top = mask_center - (y - top)
        mask_roi = gradient_mask[
            mask_top : mask_top + roi.shape[0],
            mask_left : mask_left + roi.shape[1],
        ]

        np.maximum(roi, mask_roi, out=roi)

    return map_temp


def main():
    rclpy.init()
    node = Node("data")
    scan_data_path = "./data/sim/raw_data.npz"
    map_info = read_map_yaml("./data/sim/map.yaml")
    map_raw = cv2.imread(map_info["image"], cv2.IMREAD_GRAYSCALE)
    map_raw = process_map(map_raw)
    # cv2.imshow("map", map_raw)
    # cv2.waitKey(0)
    lidar_x, lidar_y, lidar_yaw = 200, 200, 0

    while True:
        scan = wait_for_message(node, LaserScan, "/scan")
        scan_msg = {
            "angle_min": -math.pi,
            "angle_max": math.pi,
            "angle_increment": math.pi / 320,
            "range_min": 0.1,
            "range_max": 12.0,
            "ranges": scan.ranges,
        }
        scan_points = process_scan(scan_msg, map_info)
        while True:
            best_dx, best_dy, best_dyaw = calculate_matching_value(
                scan_points, lidar_x, lidar_y, lidar_yaw, map_raw
            )
            lidar_x += best_dx
            lidar_y += best_dy
            lidar_yaw += best_dyaw
            if check_position(lidar_x, lidar_y, lidar_yaw):
                break
            if abs(best_dx) < 1 and abs(best_dy) < 1 and abs(best_dyaw) < 0.01:
                break
        print(
            f"{(lidar_y - 200) * map_info['resolution']:.2f}",
            f"{(lidar_x - 200) * map_info['resolution']:.2f}",
        )


if __name__ == "__main__":
    main()
