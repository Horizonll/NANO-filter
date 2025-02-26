import autograd.numpy as np
from autograd import jacobian
from .model import Model
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan


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
        rclpy.init()
        self.node = Node("data")
        self.map = self.wait_for_message(self.node, OccupancyGrid, "/map")

    def f(self, x, u):
        odom = self.wait_for_message(self.node, Odometry, "/odom")
        return np.array(
            [
                odom.pose.pose.position.x,
                odom.pose.pose.position.y,
                odom.pose.pose.orientation.z,
            ]
        )

    def h(self, x):
        scan = self.wait_for_message(self.node, LaserScan, "/scan")

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
        return jacobian(lambda x: self.f(x))(x_hat)

    def jac_h(self, x_hat, u=0):
        return jacobian(lambda x: self.h(x))(x_hat)

    def wait_for_message(self, node, topic_type, topic):
        class _vfm(object):
            def __init__(self) -> None:
                self.msg = None

            def cb(self, msg):
                self.msg = msg

        vfm = _vfm()
        subscription = node.create_subscription(topic_type, topic, vfm.cb, 1)
        while rclpy.ok():
            if vfm.msg != None:
                return vfm.msg
            rclpy.spin_once(node)
        subscription.destroy()
