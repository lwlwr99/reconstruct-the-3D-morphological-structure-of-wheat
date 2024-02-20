import numpy as np
import math

def load_data(path):
    points = np.loadtxt(path, delimiter=' ')
    leaf_num = points[0][0]
    j_num = points[0][1]
    s_num = points[0][2]

    leaf_dict = {}
    leaf_dict_pointnum = {}
    leaf_count = 0
    j_dict = {}
    j_dict_pointnum = {}
    j_count = 0
    s_dict = {}
    s_dict_pointnum = {}
    s_count = 0

    leafs = np.ones((int(leaf_num),15000,3))
    js = np.ones((int(j_num),50000,3))
    ss = np.ones((int(s_num),5000,3))
    for i in range(1,points.shape[0]):
        if int(points[i][6]) == 100: #类别为叶片
            try:
                leaf_dict[points[i][7]] #实例号
                leaf_dict_pointnum[leaf_dict[points[i][7]]] = leaf_dict_pointnum[leaf_dict[points[i][7]]] + 1 #实例号对应点云数量
            except:
                leaf_dict[points[i][7]] = leaf_count
                leaf_dict_pointnum[leaf_dict[points[i][7]]] = 1
                leaf_count = leaf_count + 1
            leafs[leaf_dict[points[i][7]]][leaf_dict_pointnum[leaf_dict[points[i][7]]]-1][0:3] = points[i][0:3]

        elif int(points[i][6]) == 200:
            try:
                j_dict[points[i][7]]
                j_dict_pointnum[j_dict[points[i][7]]] = j_dict_pointnum[j_dict[points[i][7]]] + 1
            except:
                j_dict[points[i][7]] = j_count
                j_dict_pointnum[j_dict[points[i][7]]] = 1
                j_count = j_count + 1
            js[j_dict[points[i][7]]][j_dict_pointnum[j_dict[points[i][7]]]-1][0:3] = points[i][0:3]
            
        elif int(points[i][6]) == 300:
            try:
                s_dict[points[i][7]]
                s_dict_pointnum[s_dict[points[i][7]]] = s_dict_pointnum[s_dict[points[i][7]]] + 1
            except:
                s_dict[points[i][7]] = s_count
                s_dict_pointnum[s_dict[points[i][7]]] = 1
                s_count = s_count + 1
            ss[s_dict[points[i][7]]][s_dict_pointnum[s_dict[points[i][7]]]-1][0:3] = points[i][0:3]
    temp = np.ones((3))
    for i in range(0,leafs.shape[0]):
        for j in range(0,leafs.shape[1]):
            if leafs[i][j][0] == 1 and leafs[i][j][1] == 1 and leafs[i][j][2] == 1:
                leafs[i][j] = temp
            else:
                temp = leafs[i][j]
    for i in range(0,js.shape[0]):
        for j in range(0,js.shape[1]):
            if js[i][j][0] == 1 and js[i][j][1] == 1 and js[i][j][2] == 1:
                js[i][j] = temp
            else:
                temp = js[i][j]

    return points[1:,0:3], leafs, js, ss

#叶脉点连线
def leaf_vein_connect(pots):
    l = len(pots)
    if l <= 1:
        return [], 0
    con = [pots[0]]     # 已经连线的点集，先随便放一个点进去
    not_con = pots[1:]  # 还没连线的点集
    paths = []          # 所有连线
    length_total = 0    # 总连线长度
    begin = 0
    end = 0
    k = 0
    length_ab = 0
    for _ in range(l - 1):  # 共 l-1 条连线
        # 得到下一条连线的两点a、b 及其距离length_ab
        a, b = con[0], not_con[0]  # 先任意选两个点
        length_ab = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
        for m in con:
            for n in not_con:
                lg = math.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2 + (m[2] - n[2]) ** 2)
                if lg < length_ab:  # 如果有更短的
                    length_ab = lg
                    a, b = m, n
        # 记录首尾点云
        if k==1 and begin == pots.index(a):
            begin = pots.index(b)
        if pots.index(a)==0:
            if k==0:
                k = k + 1
            elif k==1:
                begin = pots.index(b)
        if end == pots.index(a):
            end = pots.index(b)

        paths.append([pots.index(a), pots.index(b)])   # 记录连线ab
        con.append(b)      # 已连接点集中记录点b
        not_con.remove(b)  # 未连接点集中删除点b
        length_total += length_ab  # 记录总长度
    length_total += length_ab
 
    return paths, length_total, begin, end

#obb包围盒长宽高计算
def obb_length(obb):
  points = np.asarray(obb.get_box_points())
  bottom_indices = np.argsort(points[:,2])[:4]
  top_indices = np.argsort(points[:,2])[-4:]
  length = np.linalg.norm(points[bottom_indices[0]] - points[bottom_indices[2]])
  width = np.linalg.norm(points[bottom_indices[0]] - points[bottom_indices[1]])
  height = np.linalg.norm(points[bottom_indices[0]] - points[top_indices[0]])
  return np.array([length, width, height])

#计算两点距离
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

#计算方位角和叶倾角
def calculate_angles(point1, point2):
    # 计算方向向量
    dx, dy, dz = point2[0][0] - point1[0][0], point2[0][1] - point1[0][1], point2[0][2] - point1[0][2]
    # 计算倾斜角 (与水平面的夹角)
    L = math.sqrt(dx**2 + dy**2)
    tilt_angle = math.degrees(math.atan2(abs(dz), L))
    # 计算方位角 (与北方的夹角)
    azimuth_angle = math.degrees(math.atan2(dy, dx))
    # 将方位角转换为0到360度
    if azimuth_angle < 0:
        azimuth_angle += 360
    return tilt_angle, azimuth_angle

#寻找最近的点
def find_nearest_point(cloud, target_point):
    distances = np.linalg.norm(np.asarray(cloud.points) - target_point, axis=1)
    nearest_idx = np.argmin(distances)
    nearest_point = cloud.points[nearest_idx]
    return nearest_point
#寻找点的index
def find_point_index(point_cloud, target_point):
    # Convert the target point to a numpy array
    target_np = np.array(target_point)
    # Convert the point cloud to numpy array
    points_np = np.asarray(point_cloud.points)
    # Calculate the Euclidean distance between target point and all points in the point cloud
    distances = np.linalg.norm(points_np - target_np, axis=1)
    # Find the index of the point with the minimum distance to the target point
    min_index = np.argmin(distances)
    return min_index
def find_value_row_index(matrix, target_value):
    for i, row in enumerate(matrix):
        if target_value in row:
            return i
    return None



def convert_edge_to_face(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        outfile.write("ply\n")
        outfile.write("format ascii 1.0\n")
        outfile.write("comment Created by Open3D\n")

        # Find the indices of "element vertex" and "element edge"
        vertex_index = None
        edge_index = None
        for i, line in enumerate(lines):
            if line.startswith("element vertex"):
                vertex_index = int(line.split()[2])
            elif line.startswith("element edge"):
                edge_index = int(line.split()[2])
        
        outfile.write("element vertex {}\n".format(vertex_index))
        outfile.write("property float x\n")
        outfile.write("property float y\n")
        outfile.write("property float z\n")
        outfile.write("element face {}\n".format(edge_index))
        outfile.write("property list uchar int vertex_indices\n")
        outfile.write("end_header\n")
        
        # Write vertex information
        for line in lines[11 : 11 + vertex_index]:
            outfile.write(line)

        # Write face information
        for i in range(edge_index):
            vertex1, vertex2 = map(int, lines[11 + vertex_index + i].split())
            # Add 3 to vertex indices and add the first vertex
            face_line = "3 {} {} {}\n".format(vertex1, vertex2 , vertex1)
            outfile.write(face_line)

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功") 
