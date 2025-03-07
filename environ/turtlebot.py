import autograd.numpy as np
from autograd import jacobian
from .model import Model
import math


class TurtleBot(Model):
    def __init__(
        self,
        state_outlier_flag=False,
        measurement_outlier_flag=False,
        noise_type="Gaussian",
    ):
        super().__init__(self)
        self.dim_x = 3
        self.dim_y = 3
        self.dt = 1.0 / 15
        self.x0 = np.array([0.0, 0.0, 0.0])
        self.P0 = np.diag(np.array([0.0001, 0.0001, 0.0001])) ** 2
        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0
        self.process_std = np.array([0.0034, 0.0056, 0.0041])
        self.observation_std = np.array(
            [0.0238, 0.0284, 0.0259, 0.0107, 0.0094, 0.0118]
        )
        self.obs_var = np.ones(self.dim_y) * 0.01
        self.Q = np.diag(self.process_std**2)
        self.R = np.diag(self.observation_std**2)
        self.map_info = self.read_map_yaml("./data/sim/map.yaml")

    def f(self, x, u):
        v = (u[0] + u[1]) / 2
        w = (u[0] - u[1]) / 0.233
        return np.array(
            [
                x[0] + v * np.cos(x[2]) * self.dt,
                x[1] + v * np.sin(x[2]) * self.dt,
                x[2] + w * self.dt,
            ]
        )

    def h(self, x, scan):
        scan_msg = {
            "angle_min": -math.pi,
            "angle_max": math.pi,
            "angle_increment": math.pi / 320,
            "range_min": 0.16,
            "range_max": 12.0,
            "ranges": scan,
        }
        lidar_x, lidar_y, lidar_yaw = x[0], x[1], x[2]
        scan_points = self.process_scan(scan_msg)
        while True:
            best_dx, best_dy, best_dyaw = self.calculate_matching_value(
                scan_points, lidar_x, lidar_y, lidar_yaw, self.map_raw
            )
            lidar_x += best_dx
            lidar_y += best_dy
            lidar_yaw += best_dyaw
            if abs(best_dx) < 1 and abs(best_dy) < 1 and abs(best_dyaw) < 0.01:
                break
        return np.array(
            [
                (lidar_y - 200) * self.map_info["resolution"],
                (lidar_x - 200) * self.map_info["resolution"],
                lidar_yaw,
            ]
        )

    def read_map_yaml(self, yaml_file):
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

    def process_scan(self, scan_msg):
        angles = np.arange(
            scan_msg["angle_min"], scan_msg["angle_max"], scan_msg["angle_increment"]
        )
        ranges = np.array(scan_msg["ranges"])
        valid_indices = (ranges >= scan_msg["range_min"]) & (
            ranges <= scan_msg["range_max"]
        )
        ranges = ranges[valid_indices]
        angles = angles[valid_indices]
        x = ranges * np.cos(angles) / self.map_info["resolution"]
        y = -ranges * np.sin(angles) / self.map_info["resolution"]
        return np.column_stack((x, y))

    def calculate_matching_value(
        self, scan_points, lidar_x, lidar_y, lidar_yaw, map_temp
    ):
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

    def create_gradient_mask(self, size):
        center = size // 2
        y, x = np.ogrid[:size, :size]
        distance = np.hypot(x - center, y - center)
        mask = np.clip(255 * (1 - distance / center), 0, 255).astype(np.uint8)
        return mask

    def process_map(self, map_raw):
        map_temp = np.zeros_like(map_raw, dtype=np.uint8)
        gradient_mask = self.create_gradient_mask(91)
        mask_center = gradient_mask.shape[0] // 2
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

    def f_withnoise(self, x, u=None):
        if self.state_outlier_flag:
            prob = np.random.rand()
            if prob <= 0.95:
                cov = self.Q  # 95%概率使用Q
            else:
                cov = 100 * self.Q  # 5%概率使用100Q
        else:
            cov = self.Q
        return self.f(x, u) + np.random.multivariate_normal(
            mean=np.zeros(self.dim_x), cov=cov
        )

    def h_withnoise(self, x):
        if self.noise_type == "Gaussian":
            if self.measurement_outlier_flag:
                prob = np.random.rand()
                if prob <= 0.9:
                    cov = self.R  # 95%概率使用R
                else:
                    cov = 100 * self.R  # 5%概率使用100R
            else:
                cov = self.R
            return self.h(x) + np.random.multivariate_normal(
                mean=np.zeros(self.dim_y), cov=cov
            )
        elif self.noise_type == "Beta":
            noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            noise = noise - np.mean(noise)
            return self.h(x) + noise
        else:
            return self.h(x) + np.random.laplace(
                loc=0, scale=self.obs_var, size=(self.dim_y,)
            )

    def jac_f(self, x_hat, u=0):
        return jacobian(lambda x: self.f(x, u))(x_hat)

    def jac_h(self, x_hat, u=0):
        return jacobian(lambda x: self.h(x))(x_hat)
