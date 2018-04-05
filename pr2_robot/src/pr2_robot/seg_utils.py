#!/usr/bin/env python

# Import modules
from sensor_stick.pcl_helper import *

# Initialize color_list
get_color_list.color_list = []

def denoise(cloud, k=50, x=1.0):
    filter = cloud.make_statistical_outlier_filter()
    filter.set_mean_k(k)
    filter.set_std_dev_mul_thresh(x)
    cloud_filtered = filter.filter()
    return cloud_filtered

def downsample(cloud, leaf=0.025):
    # Voxel Grid filter
    vox = cloud.make_voxel_grid_filter()
    vox.set_leaf_size(leaf, leaf, leaf)
    cloud_filtered = vox.filter()
    return cloud_filtered

def passthrough(cloud, ax='z', axmin=0.6, axmax=1.1):
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name(ax)
    passthrough.set_filter_limits(axmin, axmax)
    cloud_filtered = passthrough.filter()
    return cloud_filtered

def ransac(cloud, dmax=0.01):
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(dmax)
    inliers, coefficients = seg.segment()

    cloud_table = cloud.extract(inliers, negative=False)
    cloud_objects = cloud.extract(inliers, negative=True)

    return cloud_table, cloud_objects

def cluster(cloud, as_list=False):
    cloudrgb = cloud #save
    cloud = XYZRGB_to_XYZ(cloud)
    tree = cloud.make_kdtree()
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(2000)
    ec.set_SearchMethod(tree)
    c_idx = ec.Extract()
    #print np.shape(c_idx)
    #print np.max(c_idx)
    n_c = len(c_idx)#1 + np.max(c_idx)
    c_col = get_color_list(n_c)

    if as_list:
        # return each cloud independently
        return [cloudrgb.extract(i) for i in c_idx]

    cloud = np.float32(cloud)
    res = []
    for i in range(n_c):
        #print c_idx[i]
        #print np.int32(c_idx[i])
        ci = cloud[np.int32(c_idx[i])] # (n, 3)
        m = len(ci)
        col = rgb_to_float(c_col[i])
        col = np.full([m,1], col) # vector
        ci = np.concatenate((ci,col), axis=1) #(n,4)
        res.append(ci)
    res = np.concatenate(res, axis=0) #(n_c*m_c, 4)
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_list(res)
    return cloud
