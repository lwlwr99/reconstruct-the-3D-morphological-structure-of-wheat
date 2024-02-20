import numpy as np
import open3d as o3d
import math
from copy import deepcopy
from sklearn import cluster
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os
import random
import utils
import pandas as pd


#layer代表第几层切片
def slicePoint(stem, layer,delta):
    qiepian = []
    maxz = np.max(stem[:,2])
    minz = np.min(stem[:,2])
    qiepianweizhi = minz
    for i in range(0,stem.shape[0]):
        if(stem[i][2] > qiepianweizhi + layer * delta and stem[i][2] < qiepianweizhi + layer * delta + delta):
            qiepian.append(stem[i])
    return qiepian

def is_point_inside_cylinder(point, cylinder_center, cylinder_axis, cylinder_radius, cylinder_height):
    point = np.array(point)
    cylinder_center = np.array(cylinder_center)
    cylinder_axis = np.array(cylinder_axis)
    # 计算点到圆柱轴线的投影点
    projection = np.dot(point - cylinder_center, cylinder_axis) * cylinder_axis + cylinder_center

    # 计算投影点与点之间的距离
    distance = np.linalg.norm(point - projection)

    # 判断点是否在圆柱体内部
    if distance <= cylinder_radius and projection[2] >= cylinder_center[2] - cylinder_height/2 and projection[2] <= cylinder_center[2] + cylinder_height/2:
        return True
    else:
        return False

def build_cylinder(r,h,x1,y1,z1):
    
    # 定义圆柱参数
    radius = r  # 圆柱半径
    height = h  # 圆柱高度
    resolution = 50  # 圆柱的分辨率，即面的数量
    grid_resolution = 0.1  # 网格分辨率，控制均匀填充的密度

    # 创建圆柱点云
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)

    # 平移变换
    cylinder.translate([x1, y1, z1])
    # 获取圆柱体的边界框
    bbox = cylinder.get_axis_aligned_bounding_box()

    # 计算网格的边界
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    x_range = np.arange(min_bound[0], max_bound[0], grid_resolution)
    y_range = np.arange(min_bound[1], max_bound[1], grid_resolution)
    z_range = np.arange(min_bound[2], max_bound[2], grid_resolution)

    # 在每个网格单元中生成一个点
    points = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                query_point = [x, y, z]
                distance_to_bottom = -0.5
                distance_to_top = 0.5
                if is_point_inside_cylinder([x, y, z], (x1,y1,z1), (0,0,1), r, h):
                #distance_to_bottom <= radius and distance_to_top >= radius:  # 判断点在圆柱体内部
                    points.append(query_point)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def fix_stems(stem,layer,delta):
    seg_points = o3d.geometry.PointCloud()
    stem = np.unique(stems[0], axis=0)
    #找到第layer层的茎秆点云
    seg = slicePoint(stem,layer,delta)
    if(seg==[]):
        return None,None
    seg_points.points = o3d.utility.Vector3dVector(seg)
    cl,ind = seg_points.remove_statistical_outlier(nb_neighbors=8,std_ratio=2.5)
    seg_points = seg_points.select_by_index(ind)
    # -------------------密度聚类--------------------------
    labels = np.array(seg_points.cluster_dbscan(eps=0.8,               # 邻域距离
                                            min_points=8,          # 最小点数
                                            print_progress=False))  # 是否在控制台中可视化进度条
    max_label = labels.max()
    #print(f"point cloud has {max_label + 1} clusters")
    # ---------------------保存聚类结果------------------------
    pcd_clusters = o3d.geometry.PointCloud()
    seg_points_center_list = []
    for i in range(max_label + 1):
        ind = np.where(labels == i)[0]
        clusters_cloud = seg_points.select_by_index(ind)
        center = clusters_cloud.get_center()
        seg_points_center_list.append(center)
        cylinder = build_cylinder(0.3,delta + 0.1,center[0],center[1], seg_points.get_min_bound()[2] +  delta / 2)
        pcd_clusters += cylinder
    # --------------------可视化聚类结果----------------------
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd_clusters.paint_uniform_color((0,113/255,188/255))
    seg_points.paint_uniform_color((0,113/255,188/255))
    seg_points.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd_clusters,seg_points_center_list


#将list合并为一个line_set
def merge_linesets(line_sets):
    merged_points = []
    merged_lines = []
    offset = 0

    for line_set in line_sets:
        # 获取当前 LineSet 的顶点和线索引
        points = line_set.points
        lines = line_set.lines
        num_points = len(points)
        # 将当前 LineSet 的顶点加入到合并后的列表中
        merged_points.extend(points)
        # 将当前 LineSet 的线索引加入到合并后的列表中，同时根据偏移量调整索引
        merged_lines.extend([[line[0] + offset, line[1] + offset] for line in lines])
        # 更新偏移量，以便调整下一个 LineSet 的索引
        offset += num_points
    # 创建合并后的 LineSet 对象
    merged_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(merged_points),
        lines=o3d.utility.Vector2iVector(merged_lines)
    )

    return merged_line_set

def min_enclosing_circle(points):
    """
    计算包含所有点的最小圆。
    :param points: 点集，是一个 N x 2 的 numpy 数组。
    :return: (center, radius) 圆心和半径。
    """
    def mec(points):
        # 基准情况
        if len(points) == 0:
            return np.array([0, 0]), 0
        elif len(points) == 1:
            return points[0], 0
        elif len(points) == 2:
            center = (points[0] + points[1]) / 2
            radius = np.linalg.norm(points[0] - center)
            return center, radius

        # 检查一个点是否在给定的圆内
        def in_circle(point, circle):
            center, radius = circle
            return np.linalg.norm(point - center) <= radius

        # 递归求解
        circle = mec(points[:-1])
        if in_circle(points[-1], circle):
            return circle
        else:
            for i in range(len(points) - 1):
                circle = mec(np.delete(points, i, axis=0))
                if in_circle(points[-1], circle):
                    return circle
            return points[-1], 0

    return mec(np.array(points))


def calculate_angle_between_stems(stem_main, stem_tiller):
    """
    计算两个茎秆之间的夹角和方位角。

    :param stem_main: 主茎的点云数据。
    :param stem_tiller: 分蘖茎秆的点云数据。
    :return: 夹角（以度为单位）和方位角（以度为单位）。
    """
    # 为主茎和分蘖茎秆计算中间三分之一段和三分之二段的点
    third_main = len(stem_main) // 3
    third_tiller = len(stem_tiller) // 3

    point_main_1 = stem_main[third_main]
    point_main_2 = stem_main[2 * third_main]
    point_tiller_1 = stem_tiller[third_tiller]
    point_tiller_2 = stem_tiller[2 * third_tiller]

    # 用这些点拟合直线
    line_vector_main = np.array(point_main_2) - np.array(point_main_1)
    line_vector_tiller = np.array(point_tiller_2) - np.array(point_tiller_1)

    # 检查向量是否共线
    cross_product = np.cross(line_vector_main, line_vector_tiller)
    is_collinear = np.all(cross_product == 0)

    # 重新计算夹角，考虑共线情况
    if is_collinear:
        angle_deg = 0 if np.dot(line_vector_main, line_vector_tiller) > 0 else 180
    else:
        angle = np.arccos(np.dot(line_vector_main, line_vector_tiller) / (np.linalg.norm(line_vector_main) * np.linalg.norm(line_vector_tiller)))
        angle_deg = np.degrees(angle)

    # 计算方位角
    # 方位角是从北方向开始的水平角度，计算line_vector_tiller相对于北方向的角度
    north = np.array([0, 1, 0])  # 假设北方向为Y轴正方向
    azimuth = np.arccos(np.dot(line_vector_tiller[:2], north[:2]) / (np.linalg.norm(line_vector_tiller[:2]) * np.linalg.norm(north[:2])))
    azimuth_deg = np.degrees(azimuth)

    # 考虑西半球或东半球
    if line_vector_tiller[0] < 0:  # 如果X轴负方向（西方）
        azimuth_deg = 360 - azimuth_deg

    return angle_deg, azimuth_deg

def find_point_index(point_cloud, target_point):
    """
    在点云中查找特定点的下标。
    :param point_cloud: 点云数据，形式为嵌套列表。
    :param target_point: 要查找的点，形式为 [x, y, z]。
    :return: 目标点的下标，如果没有找到则返回 None。
    """
    min_distance = float('inf')
    closest_index = None
    c_i = None
    target_point_np = np.array(target_point)

    for index, point in enumerate(point_cloud):
        point_np = np.array(point)
        # 计算距离
        for i in range(0,len(point)):
            point_np = point[i]
            distance = np.linalg.norm(point_np - target_point_np)
            # 更新最近点和距离
            if distance < min_distance:
                min_distance = distance
                closest_index = index
                c_i = i
    return min_distance, c_i, closest_index

def find_point_index_ear(point_cloud, target_point):
    """
    在点云中查找特定点的下标，其中Z轴的距离权重为2倍。
    :param point_cloud: 点云数据，形式为嵌套列表。
    :param target_point: 要查找的点，形式为 [x, y, z]。
    :return: 目标点的下标，如果没有找到则返回 None。
    """
    min_distance = float('inf')
    closest_index = None
    c_i = None
    target_point_np = np.array(target_point) * [2, 2, 1]

    for index, point in enumerate(point_cloud):
        point_np = np.array(point)  * [2, 2, 1]
        # 计算加权后的距离
        for i in range(0,len(point)):
            point_np = point[i]
            distance = np.linalg.norm(point_np - target_point_np)
            # 更新最近点和距离
            if distance < min_distance:
                min_distance = distance
                closest_index = index
                c_i = i

    return min_distance, c_i, closest_index

def calculate_angle_between_lines(P1, P2, P3, P4):
    """
    计算计算茎叶夹角。

    :param P1, P2: 第一条线的两个端点。
    :param P3, P4: 第二条线的两个端点。
    :return: 两条线之间的夹角（以度为单位）。
    """
    # 计算方向向量
    A = np.array(P2) - np.array(P1)
    B = np.array(P4) - np.array(P3)

    # 计算夹角的余弦值
    cos_theta = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    # 计算夹角（以度为单位）
    theta = np.arccos(cos_theta) * 180 / np.pi

    # 计算方位角
    azimuth_angle = np.arctan2(A[1], A[0]) * 180 / np.pi
    # 调整角度，使其相对于北方
    azimuth_angle = (azimuth_angle + 360) % 360

    return theta, azimuth_angle


#读取点云
data = {}

file_name = 'wheat8.txt'
name = file_name.split('.')[0]
if not os.path.exists(name):
    os.makedirs(name)
    os.makedirs(name + "/leaf")
    os.makedirs(name + "/ear")
    os.makedirs(name + "/stem")
delta = 2 #切片间隔
point, leaves, stems, ears = utils.load_data(file_name)

wheat_pointcloud = o3d.geometry.PointCloud()
wheat_pointcloud.points = o3d.utility.Vector3dVector(point)
wheat_center_point = wheat_pointcloud.get_center()#小麦点云的中心点
wheat_height = wheat_pointcloud.get_max_bound()[2] - wheat_pointcloud.get_min_bound()[2]#株高

data["name"] = name
data["wheat_height"] = wheat_height
data["boundary"] = 0
data["leaf_number"] = leaves.shape[0]
data["ear_number"] = ears.shape[0]
data["stem_number"] = 0

data["leaf_length"] = []
data["leaf_width"] = []
data["leaf_pos"] = []
data["leaf_angle"] = []
data["leaf_azimuth_angle"] = []
data["leaf_of_stem"] = []

data["stem_length"] = []
data["stem_angle"] = []
data["stem_azimuth_angle"] = []

if ears.shape[0]>0:
    data["ear_length"] = []
    data["ear_vol"] = []
    data["ear_pos"] = []

print("株高：" + str(wheat_height))
print("叶数：" + str(leaves.shape[0]))
print("麦穗数：" + str(ears.shape[0]))


leaf_vein_line = []
leaf_end_points_list = []#叶基部点
leaf_begin_points_list = []#叶尖部点
leaf_end_before_points_list = []#叶基部点前一个
leaf_begin_next_points_list = []#叶尖部点下一个


#叶片分割
for j in range(0,leaves.shape[0]):
    #除去0数组  leaves中有全0的空数组
    leaf_point = np.unique(leaves[j], axis=0)
    if(leaf_point.shape[0] < 10):
        continue

    leaf_pointcloud = o3d.geometry.PointCloud()
    leaf_pointcloud.points = o3d.utility.Vector3dVector(leaf_point)
    o3d.io.write_point_cloud(name + "/leaf/leaf%d.pcd"%j, leaf_pointcloud)


    #根据点云数量确定叶片聚类段数
    if leaf_point.shape[0] < 1000: # 聚类簇数
        n_clusters = 4
    elif leaf_point.shape[0] > 1000:
        n_clusters = 7

    #将叶片分割成片段
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(leaf_point)
    leaf_kmeans_result = kmeans.predict(leaf_point)

    #将每个聚类片段保存，并记录片段质心
    leaf_kmeans_fragments = []
    leaf_kmeans_centers = []
    for i in range(0,n_clusters):
        leaf_kmeans_fragments.append(o3d.geometry.PointCloud())
        idx = np.where(leaf_kmeans_result == i)[0]
        leaf_kmeans_fragments[i] = leaf_pointcloud.select_by_index(idx)
        leaf_kmeans_centers.append(leaf_kmeans_fragments[i].get_center().tolist())
    labels = kmeans.labels_
    random_colors = np.random.rand(n_clusters, 3)
    leaf_pointcloud.colors = o3d.utility.Vector3dVector(random_colors[labels]) #根据分割结果附上颜色

    paths, length, begin, end = utils.leaf_vein_connect(leaf_kmeans_centers)
    p = [0]#连线顺序 p[i]表示第i个位置的片段是leaf_kmeans_centers[i]
    for i in paths:
        if i[0] == p[-1]:
            p.append(i[1])
        elif i[0] == p[0]:
            p.insert(0,i[1])

    #寻找两个端点
    max_distance = 0
    head_points = np.array(leaf_kmeans_fragments[p[0]].points)#边缘片段 假定是叶尖
    head_next_point = leaf_kmeans_fragments[p[1]].get_center()#边缘片段的邻居片段中心点
    head_point_index = 0#叶尖点在边缘片段中点的下标号
    #计算在headpoints中，距离headnextpoints中心点
    for i in range(0,head_points.shape[0]):
        distance = np.linalg.norm(head_points[i] - head_next_point)
        if max_distance < distance:
            max_distance = distance
            head_point_index = i

    max_distance = 0
    end_points = np.array(leaf_kmeans_fragments[p[-1]].points)
    end_before_point = leaf_kmeans_fragments[p[-2]].get_center()
    end_point_index = 0
    for i in range(0,end_points.shape[0]):
        distance = np.linalg.norm(end_points[i] - end_before_point)
        if max_distance < distance:
            max_distance = distance
            end_point_index = i

    head_point = head_points[head_point_index]
    end_point = end_points[end_point_index]
    leaf_kmeans_centers.append(head_point.tolist())
    leaf_kmeans_centers.append(end_point.tolist())

    paths, length, begin, end = utils.leaf_vein_connect(leaf_kmeans_centers)
    print("叶长：" + str(length))
    data["leaf_length"].append(length)

    #提取叶片中间片段  计算叶宽
    center_fragments = leaf_kmeans_fragments[p[int(len(p)/2)]] + leaf_kmeans_fragments[p[int(len(p)/2)-1]]
    obb = center_fragments.get_oriented_bounding_box()
    width = utils.obb_length(obb)[1]
    print("叶宽：" + str(width))
    data["leaf_width"].append(width)
    
    #叶脉点
    leaf_vein_points = o3d.geometry.PointCloud()
    leaf_vein_points.points = o3d.utility.Vector3dVector(leaf_kmeans_centers)
    leaf_vein_points.paint_uniform_color((1,0,0))
    #绘制叶脉折线
    leaf_vein_line.append(o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(leaf_kmeans_centers),
        lines=o3d.utility.Vector2iVector(paths),
    ))
    #保存叶片 叶脉曲线 叶脉曲线上的点
    leaf_name =  name + "/leaf/leaf_seg%d.pcd"%j
    leaf_ply_edge = name + "/leaf/leaf_in_%d.ply"%j
    leaf_ply_face = name + "/leaf/leaf_out_%d.ply"%j
    leaf_veinpoints_pcd_name = name + "/leaf/leaf_center_%d.pcd"%j
    o3d.io.write_point_cloud(leaf_name, leaf_pointcloud)
    o3d.io.write_line_set(leaf_ply_edge, leaf_vein_line[j], write_ascii=True)
    o3d.io.write_point_cloud(leaf_veinpoints_pcd_name, leaf_vein_points)
    utils.convert_edge_to_face(leaf_ply_edge, leaf_ply_face)#输入的ply只是线，不能在cloucompare中显示，要转成面片

    #计算两个端点与整个小麦中心点是水平距离
    dis1 = pow(leaf_kmeans_centers[p[-2]][0] - wheat_center_point[0], 2) + pow(leaf_kmeans_centers[p[-2]][1] - wheat_center_point[1], 2)
    dis2 = pow(leaf_kmeans_centers[p[-1]][0] - wheat_center_point[0], 2) + pow(leaf_kmeans_centers[p[-1]][1] - wheat_center_point[1], 2)

    endpoint = []
    endpoint2 = []

    wheat_min_z = wheat_pointcloud.get_min_bound()[2]

    if(abs(dis1-dis2)<3 or (head_point[2] - wheat_min_z > wheat_height/3 and end_point[2] - wheat_min_z > wheat_height/3)):
        if(head_point[2] > end_point[2]):
            leaf_kmeans_fragments[p[-1]].paint_uniform_color((1,0,0))
            leaf_end_points_list.append(end_point)
            leaf_begin_points_list.append(head_point)

            leaf_end_before_points_list.append(end_before_point)
            leaf_begin_next_points_list.append(head_next_point)

            endpoint.append(end_point)
            endpoint2.append(end_before_point)
        else:
            leaf_kmeans_fragments[p[0]].paint_uniform_color((1,0,0))
            leaf_end_points_list.append(head_point)
            leaf_begin_points_list.append(end_point)

            leaf_end_before_points_list.append(head_next_point)
            leaf_begin_next_points_list.append(end_before_point)

            endpoint.append(head_point)
            endpoint2.append(head_next_point)
    else:
        if(dis1 > dis2):
            leaf_kmeans_fragments[p[-1]].paint_uniform_color((1,0,0))
            leaf_end_points_list.append(end_point)
            leaf_begin_points_list.append(head_point)

            leaf_end_before_points_list.append(end_before_point)
            leaf_begin_next_points_list.append(head_next_point)

            endpoint.append(end_point)
            endpoint2.append(end_before_point)
        else:
            leaf_kmeans_fragments[p[0]].paint_uniform_color((1,0,0))
            leaf_end_points_list.append(head_point)
            leaf_begin_points_list.append(end_point)

            leaf_end_before_points_list.append(head_next_point)
            leaf_begin_next_points_list.append(end_before_point)
            
            endpoint.append(head_point)
            endpoint2.append(head_next_point)
    

leaf_end_points = o3d.geometry.PointCloud()
leaf_end_points.points = o3d.utility.Vector3dVector(leaf_end_points_list)
leaf_end_points.paint_uniform_color((1,0,0))
o3d.io.write_point_cloud(name + "/leaf_end_points.pcd", leaf_end_points, write_ascii=True)
#o3d.visualization.draw_geometries([leaf_end_points], window_name="1")

stems_points = o3d.geometry.PointCloud()
stems_points.points = o3d.utility.Vector3dVector(stems[0])
stems_points.paint_uniform_color((0,113/255,188/255))
o3d.io.write_point_cloud(name + "/stem/stem.pcd", stems_points, write_ascii=True)

stem_height_1_3 = []

cylinder_stems = o3d.geometry.PointCloud()
stem_seg_num = int(wheat_height/delta)
cylinder_centers = []
for i in range(0,stem_seg_num):
    points,center = fix_stems(stems_points,i,delta)#单个切片点云和单个切片点云的聚类后中心点
    if points == None:
        continue
    cylinder_stems += points
    if center != []:
        cylinder_centers.append(center)
    if i == stem_seg_num // 2:
        stem_height_1_3 = center

cylinder_stems.paint_uniform_color((0,113/255,188/255))


point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(stem_height_1_3)

# 计算凸包
hull, _ = point_cloud.compute_convex_hull()
hull_points = hull.vertices
center, radius = min_enclosing_circle(hull_points)
center[2] = stem_seg_num // 2 * delta

data["boundary"] = radius
print("外接圆半径：" + str(radius))

z_value = wheat_height // 2  # 假设所有点都在同一平面上
print(z_value)

num_circle_points = 100  # 圆上的点数
circle_points = np.array([
    [
        center[0] + np.cos(2 * np.pi / num_circle_points * i) * radius,
        center[1] + np.sin(2 * np.pi / num_circle_points * i) * radius,
        z_value
    ]
    for i in range(num_circle_points)
])

# 将凸包点和圆点转换为Open3D点云
hull_point_cloud = o3d.geometry.PointCloud()
hull_point_cloud.points = o3d.utility.Vector3dVector(hull_points)
circle_point_cloud = o3d.geometry.PointCloud()
circle_point_cloud.points = o3d.utility.Vector3dVector(circle_points)

o3d.io.write_point_cloud(name + "/circumscribedcircle.pcd", circle_point_cloud)


cylinder_centers_points = o3d.geometry.PointCloud()
cylinder_centers_points.points = o3d.utility.Vector3dVector(np.concatenate(cylinder_centers))

point_num = np.asarray(cylinder_centers_points.points).shape[0]#茎圆柱中心点数量
np_stem_center_points = np.asarray(cylinder_centers_points.points)
stem_center_points = np_stem_center_points[np.argsort(np_stem_center_points[:,2])]
stem_center_points = stem_center_points[::-1]#倒序 从大到小
#尾部点从上到下排序
stem_center_points_pcd = o3d.geometry.PointCloud()
stem_center_points_pcd.points = o3d.utility.Vector3dVector(stem_center_points)
stem_center_points_pcd.colors = o3d.utility.Vector3dVector(np.random.rand(point_num,3))
o3d.io.write_point_cloud(name + "/stem_center_points_pcd.pcd", stem_center_points_pcd, write_ascii=True)

horz = np.zeros((point_num,point_num),dtype=float)
vert = np.zeros((point_num,point_num),dtype=float)
score = np.zeros((point_num,point_num),dtype=float)

for i in range(0,point_num):
    for j in range(i+1,point_num):
        x1 = stem_center_points[i][0]
        y1 = stem_center_points[i][1]
        z1 = stem_center_points[i][2]
        x2 = stem_center_points[j][0]
        y2 = stem_center_points[j][1]
        z2 = stem_center_points[j][2]
        
        horz[i][j] = math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
        vert[i][j] = abs(z1-z2)
        score[i][j] = horz[i][j] * 2.0 + vert[i][j] * 0.2

stem_connect = []
traversed = []
connection_order = []

for i in range(0,point_num):
    mins = 999
    mins_index = 0
    for j in range(i+1,point_num):
        if score[i][j] != 0:
            if(vert[i][j]/horz[i][j] > 1.8):#两点之间倾斜角度不能过大
                if(score[i][j] < mins and horz[i][j] < 3):#分数低 并且垂直距离不过大
                    mins = score[i][j]
                    mins_index = j
    connection_order.append(mins_index)

for i in np.arange(point_num):#遍历所有的茎秆点
    if not (i in traversed):#如果没有遍历到
        traversed.append(i)#标记已遍历
        stem_connect.append([i])#是茎的最高点

    for k in range(0,len(stem_connect)):
        if i in stem_connect[k] and not(connection_order[i] in traversed):
            stem_connect[k].append(connection_order[i])
    traversed.append(connection_order[i])

del_count = 0
for i in range(0,len(stem_connect)):
    if len(stem_connect[i - del_count]) <= 4:
        del stem_connect[i - del_count]
        del_count = del_count+1

stem_connect_points = []
for i in range(0,len(stem_connect)):
    stem_connect_points.append([])
    for j in range(0,len(stem_connect[i])):
        stem_connect_points[i].append(stem_center_points[stem_connect[i][j]].tolist())

data["stem_number"] = len(stem_connect)
print("分蘖数：" + str(len(stem_connect)))
stem_center_points_pcd.points = o3d.utility.Vector3dVector(stem_center_points)


stem_length = []
for i in range(0,len(stem_connect_points)):
    distance = 0
    for j in range(0,len(stem_connect_points[i])-1):
        distance += utils.calculate_distance(stem_connect_points[i][j], stem_connect_points[i][j+1])
    distance += stem_connect_points[i][-1][2] - wheat_min_z
    stem_length.append(distance)

    #和主茎夹角
    angle_between_stems, azimuth_between_stems = calculate_angle_between_stems(stem_connect_points[0], stem_connect_points[i])

    data["stem_length"].append(distance)
    data["stem_angle"].append(angle_between_stems)
    data["stem_azimuth_angle"].append(azimuth_between_stems)
    print("茎长：" + str(distance))
    print("分蘖与主茎夹角：" + str(angle_between_stems))
    print("分蘖方位角：" + str(azimuth_between_stems))

    stem_pcd = o3d.geometry.PointCloud()
    stem_pcd.points = o3d.utility.Vector3dVector(stem_connect_points[i])
    if(angle_between_stems==0):
        stem_pcd.paint_uniform_color((1,0,0))
    else:
        stem_pcd.paint_uniform_color((0,113/255,188/255))
    o3d.io.write_point_cloud(name + "/stem/stem%d.pcd"%i, stem_pcd)


#叶片归到分蘖 并纠正叶基点

distances_before = []#基点和对应最近骨架点的距离
distances_after = []
all_distance_before = 0
for i in range(0,len(leaf_end_points_list)):
    #叶基
    distance1, c_i1, index1 = find_point_index(stem_connect_points, leaf_end_points_list[i])
    #叶尖
    distance2, c_i2, index2 = find_point_index(stem_connect_points, leaf_begin_points_list[i])
    distances_before.append(distance1)
    distances_after.append(distance2)
    all_distance_before += distance1

for i in range(0,len(distances_before)):
    if distances_before[i] > 1.75 * all_distance_before/len(distances_before):
        if distances_before[i] > distances_after[i]:
            temp = leaf_end_points_list[i]
            leaf_end_points_list[i] = leaf_begin_points_list[i]
            leaf_begin_points_list[i] = temp
            temp = leaf_end_before_points_list[i]
            leaf_end_before_points_list[i] = leaf_begin_next_points_list[i]
            leaf_begin_next_points_list[i] = temp

distances = []
leaf_stem_list = []#叶片对应的茎序号
close_stem_points_index = []#最近点在该茎上的编号
closest_stem_point = []#叶片基点和茎秆骨架点最近点
for i in range(0,len(leaf_end_points_list)):
    distance, c_i, index = find_point_index(stem_connect_points, leaf_end_points_list[i])
    distances.append(distance)
    leaf_stem_list.append(index)
    close_stem_points_index.append(c_i)
    closest_stem_point.append(stem_connect_points[index][c_i])

    leaf_end_point1 = leaf_end_points_list[i]
    leaf_end_point2 = leaf_end_before_points_list[i]
    
    if(c_i-2 < 0):
        stem_point2 = stem_connect_points[index][c_i]
    else:
        stem_point2 = stem_connect_points[index][c_i-2]

    if(c_i+2 > len(stem_connect_points[index])-1):
        stem_point1 = stem_connect_points[index][c_i]
    else:
        stem_point1 = stem_connect_points[index][c_i+2]
    
    angle, azimuth_angle = calculate_angle_between_lines(leaf_end_point1, leaf_end_point2, stem_point1, stem_point2)

    data["leaf_pos"].append(leaf_end_point1)
    data["leaf_angle"].append(angle)
    data["leaf_azimuth_angle"].append(azimuth_angle)
    data["leaf_of_stem"].append(index)
    print("茎叶夹角：" + str(angle))
    print("叶片方位角：" + str(azimuth_angle))


average_distance = sum(distances) / len(distances)
print("平均距离：")
print(average_distance)
print("叶片对应茎秆：")
print(leaf_stem_list)

leaf_stem_close = o3d.geometry.PointCloud()
leaf_stem_close.points = o3d.utility.Vector3dVector(closest_stem_point)
leaf_stem_close.paint_uniform_color((1,0,0))
o3d.io.write_point_cloud(name + "/leaf_stem_close.pcd", leaf_stem_close)
leaf_end_points.points = o3d.utility.Vector3dVector(leaf_end_points_list)
leaf_end_points.paint_uniform_color((1,0,0))
o3d.io.write_point_cloud(name + "/leaf_end_points_fix.pcd", leaf_end_points, write_ascii=True)

ear = o3d.geometry.PointCloud()
ear_pos_list = []
for j in range(0,ears.shape[0]):
    mask = ~np.all(ears[j] == [1, 1, 1], axis=1)
    ear_point = ears[j][mask]
    ear_pcd = o3d.geometry.PointCloud()
    ear_pcd.points = o3d.utility.Vector3dVector(ear_point)

    if len(ear_point)==0:
        continue

    obb = ear_pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)

    # 获取OBB的8个顶点
    vertices = np.asarray(obb.get_box_points())

    # 定义OBB的12个三角形的顶点索引
    triangles = [
        [0, 1, 0], [1, 7, 1],  # 底面
        [7, 2, 7], [2, 0, 2],  # 顶面
        [3, 6, 3], [6, 4, 6],  # 前面
        [4, 5, 4], [5, 3, 5],  # 后面
        [0, 3, 0], [1, 6, 1],  # 左面
        [7, 4, 7], [2, 5, 2]   # 右面
    ]

    # 创建三角网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # 保存网格为PLY文件
    o3d.io.write_point_cloud(name + "/ear/ear%d.pcd"%j, ear_pcd)
    o3d.io.write_triangle_mesh(name + "/ear/ear%dbox.ply"%j, mesh)

    #utils.convert_edge_to_face("ear%d_input.ply"%j, "ear%d.ply"%j)
    length = max(utils.obb_length(obb))
    data["ear_length"].append(length)
    print("麦穗长：" + str(length))

    ear_position = min(ear_pcd.points, key=lambda point: point[2])
    data["ear_pos"].append(ear_position)
    print("麦穗着生点：" + str(ear_position))
    ear_pos_list.append(ear_position)

    hull = ConvexHull(ear_point)
    # 凸包体积
    volume = hull.volume
    data["ear_vol"].append(volume)
    print(f"凸包体积: {volume}")


ear.points = o3d.utility.Vector3dVector(ear_pos_list)
o3d.io.write_point_cloud(name + "/ear_pos.pcd", ear, write_ascii=True)

stem_lines_list = []
for i in stem_connect:
    for j in range(0,len(i)-1):
        stem_lines_list.append([i[j],i[j+1]])

stem_lines = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(stem_center_points),
    lines=o3d.utility.Vector2iVector(stem_lines_list),
)

all_line = leaf_vein_line.copy()
all_line.append(stem_lines)
all_line = merge_linesets(all_line)

# 写入CSV文件
df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

# 保存为Excel文件
df.to_excel(name + '/result.xlsx', index=False, engine='openpyxl')


o3d.io.write_line_set(name + "/stem_line_input.ply", stem_lines, write_ascii=True)
utils.convert_edge_to_face(name + "/stem_line_input.ply", name + "/stem_line.ply")
o3d.io.write_line_set(name + "/all_line_input.ply", all_line, write_ascii=True)
utils.convert_edge_to_face(name + "/all_line_input.ply", name + "/all_line.ply")
