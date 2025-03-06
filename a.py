import open3d as o3d
import numpy as np
from skimage import io
import yaml
import copy


def pgm_to_point_cloud(pgm_file, resolution, origin):
    image = io.imread(pgm_file)
    points = [
        [-(x + origin[0]) * resolution, (y + origin[1]) * resolution, 0]
        for y in range(image.shape[0])
        for x in range(image.shape[1])
        if image[y, x] < 0.3
    ]
    return np.array(points)


def lidar_to_point_cloud(scan_data, angle_min, angle_max):
    angles = np.linspace(angle_min, angle_max, len(scan_data))
    points = [
        [r * np.cos(theta), r * np.sin(theta), 0] for r, theta in zip(scan_data, angles)
    ]
    return np.array(points)


def apply_icp(source, target, threshold=0.02, max_iteration=1000):
    trans_init = np.identity(4)
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.cpu.pybind.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1,
        ),
    )


def compute_normals(point_cloud, radius=0.01, max_nn=30):
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )


def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    # source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp], width=800, height=800, mesh_show_back_face=False
    )


def load_data(map_yaml_path, scan_data_path):
    with open(map_yaml_path, "r") as file:
        map_config = yaml.safe_load(file)
    pgm_file = map_config["image"]
    resolution = map_config["resolution"]
    origin = map_config["origin"]

    scan_data = np.load(scan_data_path)["scan"][0]

    return pgm_file, resolution, origin, scan_data


def delete_zero(pcd):
    # 将点云转为 numpy 数组
    points = np.asarray(pcd.points)

    # 找到非0的点
    non_zero_indices = np.where(np.any(points != 0, axis=1))[0]

    # 根据这些索引筛选点
    filtered_points = points[non_zero_indices]

    # 更新点云对象
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return pcd


def main():
    map_yaml_path = "./data/sim/map.yaml"
    scan_data_path = "./data/sim/raw_data.npz"

    pgm_file, resolution, origin, scan_data = load_data(map_yaml_path, scan_data_path)

    map_points = pgm_to_point_cloud(pgm_file, resolution, origin)
    scan_points = lidar_to_point_cloud(scan_data, -np.pi, np.pi)

    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points)

    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = o3d.utility.Vector3dVector(scan_points)
    # scan_pcd = scan_pcd.voxel_down_sample(voxel_size=1)
    map_pcd = map_pcd.voxel_down_sample(voxel_size=0.2)
    # compute_normals(map_pcd)
    # compute_normals(scan_pcd)
    scan_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=30)
    )
    map_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=30)
    )
    init_guess = np.eye(4)
    threshold = 0.001
    # icp_result = apply_icp(scan_pcd, map_pcd, threshold, initial_transformation)
    # print(icp_result)
    # lidar_initial_position = np.array([0, 0, 0, 1])
    # lidar_transformed_position = np.dot(
    #     gicp_result.transformation, lidar_initial_position
    # )
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     scan_pcd,
    #     map_pcd,
    #     threshold,
    #     init_guess,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    # )
    draw_registration_result(scan_pcd, map_pcd)
    # map_pcd = delete_zero(map_pcd)
    # scan_pcd = delete_zero(scan_pcd)
    # trans_init = reg_p2p.transformation
    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #     scan_pcd, map_pcd, threshold, trans_init
    # )
    # generalized_icp = o3d.pipelines.registration.registration_generalized_icp(
    #     scan_pcd,
    #     map_pcd,
    #     threshold,
    #     trans_init,
    #     o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=350000),
    # )

    # scan_pcd.transform(generalized_icp.transformation)

    # draw_registration_result(scan_pcd, map_pcd, generalized_icp.transformation)


if __name__ == "__main__":
    main()
