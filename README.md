数据采集

```bash
ros2 bag record /scan /odom /sim_ground_truth_pose /wheel_vels
```

数据回放

```bash
ros2 bag play data
```

两轮差速，轮距 23.3cm

数据
scan
odom_pose
odom_ori
wheel_vels
ground_truth_pose
ground_truth
