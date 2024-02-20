#include "datatype/datatype.h"
#include <ATen/ATen.h>

#include "bfs_cluster/bfs_cluster.cu"
#include "cal_iou_and_masklabel/cal_iou_and_masklabel.cu"
#include "octree_ball_query/octree_ball_query.cu"
#include "roipool/roipool.cu"
#include "sec_mean/sec_mean.cu"
#include "voxelize/voxelize.cu"
#include "knnquery/knnquery_cuda_kernel.cu"
#include "knn_query/knn_query_cuda_kernel.cu"

template void voxelize_fp_cuda<float>(Int nOutputRows, Int maxActive,
                                      Int nPlanes, float *feats,
                                      float *output_feats, Int *rules,
                                      bool average);

template void voxelize_bp_cuda<float>(Int nOutputRows, Int maxActive,
                                      Int nPlanes, float *d_output_feats,
                                      float *d_feats, Int *rules, bool average);
