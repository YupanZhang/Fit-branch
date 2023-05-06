#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # np数组处理
import time  # 计算耗时
import copy
import open3d as o3d
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
import random


# In[5]:


# 读取pcd文件 
pcd = o3d.io.read_point_cloud("c://users//zhang//desktop//t3b.pcd")

# 可视化点云
o3d.visualization.draw_geometries([pcd])


# # HDBSCAN

# In[8]:


# 将点云数据转换为numpy数组
points = np.asarray(pcd.points)

# 进行聚类
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
labels = clusterer.fit_predict(points)

# 将聚类结果标记为不同的颜色
colors = plt.cm.Spectral(labels.astype(float)/np.max(labels))

# 压缩点云
scale_factor = 1
compressed_points = np.asarray(pcd.points) * np.array([scale_factor, scale_factor, 1])

# 计算每个点的kNN
k = 10
knn_indices_list = []
nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(compressed_points)
distances, indices = nbrs.kneighbors(compressed_points)
for i in range(len(pcd.points)):
    knn_indices = indices[i]
    knn_indices_list.append(knn_indices)

# 使用HDBSCAN进行聚类
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=8, allow_single_cluster=True)
labels = clusterer.fit_predict(points)

# T1 40/5
# T2 15/3
# T3 15/8

# 获取簇的数量
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# 为每个簇随机生成一个颜色
colors = np.random.rand(num_clusters+1, 3)

color_array = np.zeros((len(pcd.points), 3))

# 将同一簇中的点用同一个颜色表示
for i, label in enumerate(labels):
    if label == -1:
        color_array[i] = [255, 255, 255] # 噪声点（标记为白色）
    else:
        color_array[i] = colors[label]

# 去除噪声点
noise_idx = np.where(labels == -1)[0]
stem_clean = np.delete(compressed_points, noise_idx, axis=0)

stem_clean_cloud = o3d.geometry.PointCloud()
stem_clean_cloud.points = o3d.utility.Vector3dVector(stem_clean)
stem_clean_cloud.colors = o3d.utility.Vector3dVector(color_array[~np.isin(np.arange(len(pcd.points)), noise_idx)]) 

print("After compression and clustering:", len(np.unique(np.array(stem_clean_cloud.colors),axis = 0) ) )

# 可视化点云和聚类结果
o3d.visualization.draw_geometries([stem_clean_cloud])


# In[12]:


colors = np.array(stem_clean_cloud.colors)
# Find unique colors in pca_colors and sort them by point count
uncolors, counts = np.unique(colors, return_counts=True, axis=0)


# # Delete small (noise) clusters

# In[13]:


# Filter out clusters with less than 30 points
keep_mask = counts > 1
keep_colors = uncolors[keep_mask]
remove_colors = uncolors[~keep_mask]

# Remove points and colors of removed clusters from stem_clean_cloud
points = np.array(stem_clean_cloud.points)
colors = np.array(stem_clean_cloud.colors)
remove_mask = np.isin(colors, remove_colors).all(1)
points = points[~remove_mask]
colors = colors[~remove_mask]
stem_clean_cloud.points = o3d.utility.Vector3dVector(points)
stem_clean_cloud.colors = o3d.utility.Vector3dVector(colors)


# In[14]:


# 可视化点云和聚类结果
o3d.visualization.draw_geometries([stem_clean_cloud])


# In[15]:

# 手改聚类结果
o3d.io.write_point_cloud("c://users//zhang//desktop//test raw.pcd", stem_clean_cloud)


# # Start Branching

# In[22]:


# 读取pcd文件 "C:\Users\hhl30\Desktop\ドローン!\rain\branch-trunk angles\Tree 3 branches all.pcd"
branches = o3d.io.read_point_cloud("c://users//zhang//desktop//test raw3.pcd")


# In[70]:


# 可视化点云
o3d.visualization.draw_geometries([branches])


# # RANSAC

# In[51]:


points = np.array(branches.points)
colors = np.array(branches.colors)
uncolors = np.unique(colors, axis = 0)

x_max = np.max(points[:, 0])
y_max = np.max(points[:, 1])
z_max = np.max(points[:, 2])

x_min = np.min(points[:, 0])
y_min = np.min(points[:, 1])
z_min = np.min(points[:, 2])

# 定义线段的起点和终点
#T3:cxb = 6.85
#   cyb = -2
#   cxt = 6.8
#   cyt = -2.2
#T1:cxb = 12.76; 13.63
#         12.79; 13.68
#T2:  11.3; 13.63
#     10.98; 13.48
#start_point = [cx, cy, np.min(points[:, 2]) - 20]
#end_point = [cx, cy, np.max(points[:, 2]) + 20]


# 定义RANSAC模型
ransac = RANSACRegressor()

# 定义线段的起点和终点
#bottom center point of the trunk
cxb = 6.85
cyb = -2

#top center point of the trunk
cxt = 6.8
cyt = -2.2

#mean
cxm = (cxb + cxt)/2
cym = (cyb + cyt)/2

start_point = [cxb, cyb, np.min(points[:, 2]) - 20]
end_point = [cxt, cyt, np.max(points[:, 2]) + 20]

def pair(num_points):
    p1 = np.array([10000,0,0])
    p2 = np.array([0,0,0])
    while np.linalg.norm(p1-np.array([cxm,cym,p1[2]])) > np.linalg.norm(p2-np.array([cxm,cym,p2[2]])):
        sample = random.sample(range(num_points), 2)
        p1, p2 = cluster_points[sample]
    return p1, p2


def leng(pp1, pp2, iinn, oout):
    if np.allclose(pp1, pp2):
        return None
    e = np.array(pp1) - np.array(pp2)
    proj_iinn = e * np.dot(np.array(iinn) - np.array(pp1), e) / np.dot(e, e)
    proj_oout = e * np.dot(np.array(oout) - np.array(pp1), e) / np.dot(e, e)
    ps = proj_iinn + np.array(pp1)
    pt = proj_oout + np.array(pp1)
    l = np.linalg.norm(ps - pt)
    print(ps, pt)
    return l

def cross_norm(A, b):
    cross = np.cross(A, b)
    return np.linalg.norm(cross, axis=1)


# In[55]:


branch_lines = []
e = []
v = []
lv = [] 
ll = {}


# 定义RANSAC算法参数
inlier_threshold = .5  # 阈值
max_iterations = 5000  # 最大迭代次数

# 遍历所有的聚类
for color in uncolors:
    # 获取聚类中的点
    cluster_indices = np.all(colors == color, axis=1).nonzero()[0]
    cluster_points = points[cluster_indices]
    cluster_points = np.array(cluster_points)
    num_points = len(cluster_points)

    # 初始化RANSAC算法参数
    best_inliers = []
    best_line = None
    best_rmse = float("inf")
    inn, out = None, None

    # 进行RANSAC迭代
    for i in range(max_iterations):
        # 随机选择两个点
        #sample = random.sample(range(num_points), 2)
        p1, p2 = pair(num_points)

        # 计算选择的两个点所在的直线参数
        line_dir = p2 - p1
        line_dir /= np.linalg.norm(line_dir)
        line_ori = p1

        # 计算所有点到直线的距离
        #print("vec",cluster_points - line_ori,np.cross(cluster_points - line_ori, line_dir))
        
        diff = cluster_points - line_ori[np.newaxis, :]
        
        distances = cross_norm(diff, line_dir)
        print()
        
        #distances = np.abs(np.cross(cluster_points - line_ori, line_dir))
        inliers_indices = np.where(distances < inlier_threshold)[0]
        inliers = cluster_points[inliers_indices]
        num_inliers = len(inliers)
        inliers_dist = distances[inliers_indices]
        
        # 计算RMSE
        if num_inliers > 2:
            #rmse = np.sqrt(np.mean(np.sum((inliers - np.mean(inliers, axis=0)) ** 2, axis=1)))
            #print(inliers_dist)
            rmse = sum(inliers_dist)
        else:                                                                                    
            rmse = float("inf")

        # 更新最优解
        if num_inliers > len(best_inliers) or (num_inliers == len(best_inliers) and rmse < best_rmse):
            best_inliers = inliers
            best_line = (line_ori, line_dir)
            best_rmse = rmse
        
            dist = np.linalg.norm(inliers[:, :2] - np.array([cxm, cym]), axis=1)
            # 记录距离最短和最长的点
            #print("jjj",len(dist),len(inliers))
            inn = inliers[np.argmin(dist)]
            out = inliers[np.argmax(dist)]
            print(p1, p2, inn, out)
            length = leng(p1, p2, inn, out)
            
    # 将p1,p2分别用inn和out代替
    branch_lines.append(best_line)
    e.append(line_dir)
    v.append(p1)
    v.append(p2)
    lv.append(inn)
    lv.append(out)
    
    ll[tuple(p1)] = length
    
    print("p1, p2 (",p1, p2, ")")
    print("inn, out (",inn, out, ")")


# # Output: Angles and lengths (named as an, counts respectively)

# In[56]:


an = []
axis = np.array(end_point)-np.array(start_point)
axis_norm = np.linalg.norm(axis)
for i in range(len(branch_lines)):
    eigen = np.array(branch_lines[i][1])
    #print("E",eigen)
    #print(eigen)
    angle = np.arccos(np.dot(list(eigen), list(axis)) / (np.linalg.norm(eigen) * axis_norm ))
    angle_degrees = np.degrees(angle)
    if angle_degrees > 180:
        continue
    an.append(angle_degrees)
    
counts = list(ll.values())
print(an)
print(max(an))
print(len(an))


# # Angle distribution

# In[59]:


effect1 = np.where(np.array(an) < 180)[0]
result1 = []
for i in effect1:
    result1.append(an[i])
    
effect2 = np.where(np.array(an) >0)[0]
result2 = []
for i in effect2:
    result2.append(an[i])  

# 绘制数据分布图
plt.hist(result2, bins=30, alpha=0.5)
x = range(len(result2))
plt.xlabel('Angles')
plt.ylabel('Frequency')
plt.title('Data Distribution')
# 设置x轴的目盛值为2的倍数
#x_axis.set_ticks(range(0, 180, 5))
plt.show()


# # Length distribution

# In[63]:


# 绘制数据分布图
plt.hist(counts, bins=30, alpha=0.5)
x = range(len(counts))
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Data Distribution')
# 设置x轴的目盛值为2的倍数
#x_axis.set_ticks(range(0, 180, 5))
plt.show()


# # Angle outliers

# In[64]:


# IQRを計算
q1, q3 = np.percentile(result2, [25, 75])
iqr = q3 - q1

# 外れ値を検出
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = [x for x in result2 if x < lower_bound or x > upper_bound]

# 箱ひげ図を描画
plt.boxplot(result2, showfliers=False)  # 外れ値は描画しない
plt.title("Box plot without outliers")
plt.show()

# IQRで外れ値を検出して箱ひげ図を描画
plt.boxplot(result2)
plt.title("Box plot of θs of with outliers")
plt.scatter(np.ones(len(outliers)), outliers, color="r", marker="x")  # 外れ値を描画
plt.show()


# # Delete outliers and output final results: (angel, length)

# In[65]:


outlier_indices = [i for i, x in enumerate(result2) if x in outliers]
al = [(an[i], counts[i]) for i in range(len(an)) if i not in outlier_indices]


# In[66]:


print(al)
print("TOTLE NUMBER = ",len(al))


# # Visualize branches only

# In[67]:


e_order = np.arange(len(lv)).reshape(-1, 2)
# 创建一个Open3D LineSet对象
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(lv)
line_set.lines = o3d.utility.Vector2iVector(e_order)

# 将每个线段都着上不同的颜色
c = []
for i in range(len(an)):
    r = random.randint(0, 255) / 255.0
    g = random.randint(0, 255) / 255.0
    b = random.randint(0, 255) / 255.0
    if not ([r,g,b] in c):
        c.append([r, g, b])

line_set.colors = o3d.utility.Vector3dVector(c)

# 可视化线段
o3d.visualization.draw_geometries([line_set])
# 设置线段宽度为3.0
#line_set_line_width = 3.0
# 可视化线段
#o3d.visualization.draw_geometries([line_set.create_mesh_coordinate_frame(line_width=line_set_line_width)])
# 可视化线段
#o3d.visualization.draw_geometries([line_set])


# # Visualize the branches and the trunk

# In[68]:


# pts
lines = np.array(line_set.lines)
pts = np.array(line_set.points)
colors = np.array(line_set.colors)

trunk1 = np.array([cxb, cyb, z_min-2])
trunk2 = np.array([cxt, cyt, z_max+2])
pts = np.vstack([pts, trunk1])
pts = np.vstack([pts, trunk2])
#e_order
axis = np.array([len(lines)*2, len(lines)*2 + 1])
e_order = np.vstack([e_order, axis])

# c
c = []
for i in range(len(an)):
    r = random.randint(0, 255) / 255.0
    g = random.randint(0, 255) / 255.0
    b = random.randint(0, 255) / 255.0
    if not ([r,g,b] in c):
        c.append([r, g, b])

c = np.array(c)
black = np.array([0,0,0])
c = np.vstack([c, black])

# # Add trunk & branches
# In[69]:


# 创建一个Open3D LineSet对象
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(pts)
line_set.lines = o3d.utility.Vector2iVector(e_order)
line_set.colors = o3d.utility.Vector3dVector(c)

# 可视化线段
o3d.visualization.draw_geometries([branches, line_set])
#points = np.array(list(points).append([cx, cy, z_max]))


# In[652]:


# 可视化线段
o3d.visualization.draw_geometries([branches, line_set])


# %%
