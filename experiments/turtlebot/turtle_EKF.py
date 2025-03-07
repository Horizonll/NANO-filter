import os
import sys
import time
import argparse
import importlib

import autograd.numpy as np
from tqdm import tqdm

sys.path.append("/home/hrz/NANO-filter")
sys.path.append("../")

from filter import NANO, EKF, UKF
from environ import UGV, TurtleBot
from data_processing import load_data
from save_and_plot import calculate_rmse, save_per_exp

np.random.seed(42)


def quat_to_euler(q):
    w, x, y, z = q
    cos_pitch_cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    cos_pitch_sin_yaw = 2.0 * (x * y + w * z)
    sin_pitch = -2.0 * (x * z - w * y)
    cos_pitch = 0.0
    sin_roll_cos_pitch = 2.0 * (y * z + w * x)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)

    cos_pitch = np.sqrt(
        cos_pitch_cos_yaw * cos_pitch_cos_yaw + cos_pitch_sin_yaw * cos_pitch_sin_yaw
    )
    yaw = np.arctan2(cos_pitch_sin_yaw, cos_pitch_cos_yaw)
    if abs(sin_pitch) >= 1:
        pitch = np.sign(sin_pitch) * np.pi / 2
    else:
        pitch = np.arcsin(sin_pitch)
    roll = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    euler = np.array([roll, pitch, yaw])

    return euler


if __name__ == "__main__":
    data = np.load("./data/sim/raw_data.npz")
    scan = data["scan"]
    scan_t = data["scan_t"] / 1e9
    wheel_vel = data["wheel_vels"]
    wheel_t = data["wheel_t"] / 1e9
    pos_gt = data["ground_truth_pose"]
    ori_gt = data["ground_truth_ori"]
    time_gt = data["ground_truth_t"] / 1e9

    scan_ = []
    wheel_vel_ = []

    for t in time_gt:
        idx = np.argmin(np.abs(scan_t - t))
        scan_.append(scan[idx])
        idx = np.argmin(np.abs(wheel_t - t))
        wheel_vel_.append(wheel_vel[idx])

    scan = np.array(scan_)
    wheel_vel = np.array(wheel_vel_)

    # TODO 1: 这里把所有数据的时间戳对对齐，以pos_gt的初始时间为基准

    euler0 = quat_to_euler(ori_gt[0])
    x0 = np.array([pos_gt[0, 0], pos_gt[0, 1], euler0[2]])

    model = TurtleBot()
    model.x0 = x0
    filter = EKF(model)

    x_pred = []
    all_time = []

    for i in tqdm(range(0, wheel_t.shape[0])):
        u = wheel_vel[i]
        y = scan[i]
        time1 = time.time()
        # perform filtering
        filter.predict(u)
        # TODO 2: 这里观测模型h(x)有了，y是什么呢
        filter.update(y)
        time2 = time.time()
        x_pred.append(filter.x)
        all_time.append(time2 - time1)

    x_pred = np.array(x_pred)
    mean_time = np.mean(all_time)

    np.save("./results/turtle_ekf", x_pred)
    print("solve time: ", mean_time)
