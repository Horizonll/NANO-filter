import rclpy
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py import message_to_ordereddict
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from irobot_create_msgs.msg import WheelVels


def main():
    rclpy.init()

    # 配置存储选项
    storage_options = StorageOptions(
        uri="./data/rosbagdata.db3",  # 替换为你的 bag 文件路径
        storage_id="sqlite3",  # 通常使用 sqlite3 存储
    )

    # 配置转换选项
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 获取所有的话题信息
    topics = reader.get_all_topics_and_types()
    for topic in topics:
        print(f"Topic: {topic.name}, Type: {topic.type}")

    scan = []
    scan_t = []
    ground_truth_pose = []
    ground_truth_ori = []
    ground_truth_t = []
    odom_pose = []
    odom_ori = []
    odom_t = []
    wheel_vels = []
    wheel_t = []

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        # 根据话题类型反序列化消息
        print(f"Time: {t}, Topic: {topic}")
        if topic == "/sim_ground_truth_pose":
            msg = deserialize_message(data, Odometry)
            ground_truth_t.append(t)
            ground_truth_pose.append(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ]
            )
            ground_truth_ori.append(
                [
                    msg.pose.pose.orientation.w,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                ]
            )

        if topic == "/scan":
            msg = deserialize_message(data, LaserScan)
            scan_t.append(t)
            scan.append(msg.ranges)

        if topic == "/odom":
            msg = deserialize_message(data, Odometry)
            odom_t.append(t)
            odom_pose.append(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ]
            )
            odom_ori.append(
                [
                    msg.pose.pose.orientation.w,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                ]
            )

        if topic == "/wheel_vels":
            msg = deserialize_message(data, WheelVels)
            wheel_t.append(t)
            wheel_vels.append(
                [
                    msg.velocity_left,
                    msg.velocity_right,
                ]
            )

    scan = np.array(scan)
    scan_t = np.array(scan_t)
    ground_truth_pose = np.array(ground_truth_pose)
    ground_truth_ori = np.array(ground_truth_ori)
    ground_truth_t = np.array(ground_truth_t)
    odom_pose = np.array(odom_pose)
    odom_ori = np.array(odom_ori)
    odom_t = np.array(odom_t)
    wheel_vels = np.array(wheel_vels)
    wheel_t = np.array(wheel_t)

    all_data = {
        "scan": scan,
        "scan_t": scan_t,
        "ground_truth_pose": ground_truth_pose,
        "ground_truth_ori": ground_truth_ori,
        "ground_truth_t": ground_truth_t,
        "odom_pose": odom_pose,
        "odom_ori": odom_ori,
        "odom_t": odom_t,
        "wheel_vels": wheel_vels,
        "wheel_t": wheel_t,
    }

    np.savez("./data/raw_data.npz", **all_data)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
