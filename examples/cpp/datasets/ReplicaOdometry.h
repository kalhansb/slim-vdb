// MIT License
//
// Copyright (c) 2026 SCovox project — Replica dataloader modelled on SceneNetOdometry.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.


#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "slimvdb/utils/Utils.h"
#include "slimvdb/Config.h"

namespace datasets {

// Replica dataloader for SLIM-VDB.
//
// Expected layout (see `data/replica_niceslam/`):
//
//   <replica_root>/
//     cam_params.json                 # shared fx, fy, cx, cy, w, h, depth scale
//     <scene>/
//       color/000000.jpg              # 8-bit RGB (unused unless realtime seg is added)
//       depth/000000.png              # uint16 PNG, meters = value / cam_params.scale
//       semantic/000000.png           # uint16 PNG, per-pixel object_id (1..101, 0=void)
//       poses.txt                     # one line per frame: 16 floats, row-major 4x4
//                                     # (cam-to-world). 2000 frames per scene.
//
// If `semantic_subdir` is overridden (e.g. "semantic_sam_clip/"), labels are read
// from that folder instead of the default "semantic/". Both must be uint16 PNGs
// with class IDs in [0, NCLASSES-1].
//
// Replica has 101 nominal classes (see info_semantic.json). Build SLIM-VDB with
// SLIMVDB_NCLASSES=102 (101 + unknown/void at id 0).
template <slimvdb::Language L>
class ReplicaDataset {
public:
    using Point = Eigen::Vector3d;
    using PointCloud = std::vector<Eigen::Vector3d>;
    using SemanticLabels = std::conditional_t<L == slimvdb::CLOSED,
                                              std::vector<uint32_t>,
                                              std::vector<std::vector<float>>>;

    // Full-argument constructor (no realtime segmentation — labels are always
    // read from disk). `semantic_subdir` selects between the GT ("semantic")
    // and prediction caches (e.g. "semantic_sam_clip").
    explicit ReplicaDataset(const std::string& replica_root_dir,
                            const std::string& sequence,
                            int n_scans = -1,
                            bool apply_pose = true,
                            bool preprocess = true,
                            float min_range = 0.0F,
                            float max_range = std::numeric_limits<float>::max(),
                            const std::string& semantic_subdir = "semantic");

    // Returns (pointcloud, semantic labels, cam-to-world pose) for frame `idx`.
    [[nodiscard]] std::tuple<PointCloud, SemanticLabels, Eigen::Matrix4d> operator[](int idx) const;

    [[nodiscard]] std::size_t size() const { return depth_files_.size(); }

public:
    bool apply_pose_ = true;
    bool preprocess_ = true;
    float min_range_ = 0.0F;
    float max_range_ = std::numeric_limits<float>::max();
    // Replica is always RGB-D; flag retained for API parity with SceneNet.
    bool rgbd_ = true;
    std::filesystem::path replica_sequence_dir_;
    static constexpr uint16_t num_semantic_classes = NCLASSES;  // compile-time

private:
    std::vector<std::string> depth_files_;
    std::vector<std::string> label_files_;
    std::vector<Eigen::Matrix4d> poses_;

    // Camera intrinsics + depth scale, loaded from <root>/cam_params.json.
    float fx_ = 0.0F;
    float fy_ = 0.0F;
    float cx_ = 0.0F;
    float cy_ = 0.0F;
    int   img_w_ = 0;
    int   img_h_ = 0;
    float depth_scale_ = 6553.5F;  // depth_m = png_value / depth_scale
};

}  // namespace datasets
