// MIT License
//
// Copyright (c) 2026 SCovox project — Replica dataloader modelled on SceneNetOdometry.

#include "ReplicaOdometry.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace {

std::vector<std::string> ListFilesSorted(const fs::path& dir,
                                         const std::vector<std::string>& extensions,
                                         int n_scans) {
    std::vector<std::string> out;
    if (!fs::exists(dir)) {
        std::cerr << "ERROR: Replica dataloader: directory does not exist: " << dir << std::endl;
        std::exit(1);
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        for (const auto& ext : extensions) {
            if (entry.path().extension() == ext) {
                out.emplace_back(entry.path().string());
                break;
            }
        }
    }
    if (out.empty()) {
        std::cerr << "ERROR: Replica dataloader: no matching files in " << dir << std::endl;
        std::exit(1);
    }
    std::sort(out.begin(), out.end());
    if (n_scans > 0 && out.size() > static_cast<size_t>(n_scans)) {
        out.erase(out.begin() + n_scans, out.end());
    }
    return out;
}

// Poses: each line is 16 whitespace-separated floats, row-major cam-to-world 4x4.
// If a scene has fewer pose rows than frames, abort; if more, truncate to n_frames.
std::vector<Eigen::Matrix4d> ReadPoses(const fs::path& poses_file, size_t n_frames) {
    std::vector<Eigen::Matrix4d> poses;
    std::ifstream in(poses_file);
    if (!in.is_open()) {
        std::cerr << "ERROR: Replica dataloader: could not open poses file: " << poses_file << std::endl;
        std::exit(1);
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
        bool ok = true;
        for (int r = 0; r < 4 && ok; ++r) {
            for (int c = 0; c < 4 && ok; ++c) {
                if (!(iss >> P(r, c))) ok = false;
            }
        }
        if (ok) poses.emplace_back(P);
    }
    if (poses.size() < n_frames) {
        std::cerr << "ERROR: Replica dataloader: poses.txt has " << poses.size()
                  << " entries, need at least " << n_frames << std::endl;
        std::exit(1);
    }
    if (poses.size() > n_frames) poses.resize(n_frames);
    return poses;
}

// Tiny JSON-ish reader specialised for `cam_params.json`:
//   { "camera": { "w": 1200, "h": 680, "fx": 600.0, "fy": 600.0,
//                 "cx": 599.5, "cy": 339.5, "scale": 6553.5 } }
// Pulls numeric values by key name via regex. Avoids a JSON dependency.
struct CamParams {
    int w = 0, h = 0;
    float fx = 0.F, fy = 0.F, cx = 0.F, cy = 0.F;
    float scale = 6553.5F;
};

CamParams ReadCamParams(const fs::path& cam_params_file) {
    std::ifstream in(cam_params_file);
    if (!in.is_open()) {
        std::cerr << "ERROR: Replica dataloader: could not open cam_params: " << cam_params_file << std::endl;
        std::exit(1);
    }
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();

    auto grab = [&](const std::string& key, double fallback) -> double {
        std::regex rx("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
        std::smatch m;
        if (std::regex_search(text, m, rx)) {
            try { return std::stod(m[1].str()); } catch (...) {}
        }
        return fallback;
    };

    CamParams p;
    p.w     = static_cast<int>(grab("w", 0));
    p.h     = static_cast<int>(grab("h", 0));
    p.fx    = static_cast<float>(grab("fx", 0.0));
    p.fy    = static_cast<float>(grab("fy", 0.0));
    p.cx    = static_cast<float>(grab("cx", 0.0));
    p.cy    = static_cast<float>(grab("cy", 0.0));
    p.scale = static_cast<float>(grab("scale", 6553.5));

    if (p.fx <= 0.F || p.fy <= 0.F || p.w <= 0 || p.h <= 0) {
        std::cerr << "ERROR: Replica dataloader: cam_params missing required fields (fx, fy, w, h) in "
                  << cam_params_file << std::endl;
        std::exit(1);
    }
    return p;
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<uint32_t>>
ReadReplicaDepthAndLabels(const std::string& depth_path,
                          const std::string& label_path,
                          float fx, float fy, float cx, float cy,
                          float depth_scale,
                          float min_range, float max_range) {
    cv::Mat depth_raw = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (depth_raw.empty()) {
        std::cerr << "ERROR: could not read depth image " << depth_path << std::endl;
        std::exit(1);
    }
    if (depth_raw.type() != CV_16UC1) {
        std::cerr << "ERROR: expected uint16 depth PNG, got type " << depth_raw.type()
                  << " for " << depth_path << std::endl;
        std::exit(1);
    }

    cv::Mat label_mat = cv::imread(label_path, cv::IMREAD_UNCHANGED);
    if (label_mat.empty()) {
        std::cerr << "ERROR: could not read label image " << label_path << std::endl;
        std::exit(1);
    }
    if (label_mat.size() != depth_raw.size()) {
        std::cerr << "ERROR: depth/label size mismatch for " << depth_path << std::endl;
        std::exit(1);
    }

    // Depth in metres.
    cv::Mat depth_m;
    depth_raw.convertTo(depth_m, CV_32F, 1.0 / static_cast<double>(depth_scale));

    const int H = depth_m.rows;
    const int W = depth_m.cols;
    const float* depth_data = reinterpret_cast<const float*>(depth_m.data);

    // Labels may be CV_16UC1 or CV_8UC1 depending on upstream.
    const bool label_is_16u = (label_mat.type() == CV_16UC1);
    const uint16_t* lbl16 = label_is_16u ? reinterpret_cast<const uint16_t*>(label_mat.data) : nullptr;
    const uint8_t*  lbl8  = label_is_16u ? nullptr : reinterpret_cast<const uint8_t*>(label_mat.data);

    std::vector<Eigen::Vector3d> pts;
    std::vector<uint32_t>        labels;
    pts.reserve(static_cast<size_t>(H) * W / 4);
    labels.reserve(static_cast<size_t>(H) * W / 4);

    for (int u = 0; u < H; ++u) {
        for (int v = 0; v < W; ++v) {
            const size_t i = static_cast<size_t>(u) * W + v;
            const float d = depth_data[i];
            if (d <= 0.F || std::isnan(d) || d < min_range || d > max_range) continue;

            // Replica depth is a planar Z value (OpenCV convention), not a ray length.
            // SceneNet normalised by the ray-length factor, but NICE-SLAM renders depth
            // as planar Z — so we project directly without the sqrt normalisation.
            const float x = (static_cast<float>(v) - cx) / fx * d;
            const float y = (static_cast<float>(u) - cy) / fy * d;
            pts.emplace_back(x, y, d);
            labels.push_back(label_is_16u
                                 ? static_cast<uint32_t>(lbl16[i])
                                 : static_cast<uint32_t>(lbl8[i]));
        }
    }
    return {std::move(pts), std::move(labels)};
}

void PreProcessCloud(std::vector<Eigen::Vector3d>& points,
                     std::vector<uint32_t>& labels,
                     float min_range, float max_range) {
    size_t out = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        const double r = points[i].norm();
        if (r < min_range || r > max_range) continue;
        if (out != i) {
            points[out] = points[i];
            labels[out] = labels[i];
        }
        ++out;
    }
    points.resize(out);
    labels.resize(out);
}

void TransformPoints(std::vector<Eigen::Vector3d>& points, const Eigen::Matrix4d& T) {
    for (auto& p : points) {
        const Eigen::Vector4d q = T * Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0);
        p = q.head<3>() / q(3);
    }
}

}  // namespace

namespace datasets {

template <slimvdb::Language L>
ReplicaDataset<L>::ReplicaDataset(const std::string& replica_root_dir,
                                  const std::string& sequence,
                                  int n_scans,
                                  bool apply_pose,
                                  bool preprocess,
                                  float min_range,
                                  float max_range,
                                  const std::string& semantic_subdir)
    : apply_pose_(apply_pose),
      preprocess_(preprocess),
      min_range_(min_range),
      max_range_(max_range) {
    const fs::path root = fs::absolute(fs::path(replica_root_dir));
    replica_sequence_dir_ = root / sequence;

    const CamParams cp = ReadCamParams(root / "cam_params.json");
    fx_ = cp.fx; fy_ = cp.fy; cx_ = cp.cx; cy_ = cp.cy;
    img_w_ = cp.w; img_h_ = cp.h; depth_scale_ = cp.scale;

    depth_files_ = ListFilesSorted(replica_sequence_dir_ / "depth", {".png"}, n_scans);
    label_files_ = ListFilesSorted(replica_sequence_dir_ / semantic_subdir, {".png"}, n_scans);

    if (depth_files_.size() != label_files_.size()) {
        std::cerr << "ERROR: Replica dataloader: depth/label count mismatch ("
                  << depth_files_.size() << " vs " << label_files_.size()
                  << ") in " << replica_sequence_dir_ << std::endl;
        std::exit(1);
    }

    poses_ = ReadPoses(replica_sequence_dir_ / "poses.txt", depth_files_.size());
}

template <slimvdb::Language L>
std::tuple<std::vector<Eigen::Vector3d>,
           typename ReplicaDataset<L>::SemanticLabels,
           Eigen::Matrix4d>
ReplicaDataset<L>::operator[](int idx) const {
    std::vector<Eigen::Vector3d> points;
    SemanticLabels semantics;

    if constexpr (L == slimvdb::CLOSED) {
        std::tie(points, semantics) =
            ReadReplicaDepthAndLabels(depth_files_[idx], label_files_[idx],
                                      fx_, fy_, cx_, cy_, depth_scale_,
                                      min_range_, max_range_);
        if (preprocess_) {
            PreProcessCloud(points, semantics, min_range_, max_range_);
        }
        if (apply_pose_) {
            TransformPoints(points, poses_[idx]);
        }
    } else {
        // Open-set path is intentionally omitted: SLIM-VDB's OPEN templates
        // depend on VecXf/VecXFGrid types that upstream openvdb fork does not
        // provide (see our local patches in VDBVolume.h / common.h). Replica
        // only needs closed-set, so this branch is a fail-fast runtime guard.
        std::cerr << "ERROR: Replica dataloader only supports CLOSED-set semantics. "
                     "Rebuild SLIM-VDB with -DSLIMVDB_LANGUAGE=CLOSED." << std::endl;
        std::exit(1);
    }

    return {std::move(points), std::move(semantics), poses_[idx]};
}

template class ReplicaDataset<slimvdb::Language::CLOSED>;
// OPEN instantiation suppressed to mirror the SLIM-VDB core patch.
// template class ReplicaDataset<slimvdb::Language::OPEN>;

}  // namespace datasets
