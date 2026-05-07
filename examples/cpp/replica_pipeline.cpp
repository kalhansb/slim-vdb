// MIT License
//
// Copyright (c) 2026 SCovox project — Replica pipeline modelled on scenenet_pipeline.cpp.

#include <fmt/core.h>
#include <fmt/format.h>
#include <openvdb/openvdb.h>
#include <slimvdb/VDBVolume.h>

#include <Eigen/Geometry>
#include <argparse/argparse.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "datasets/ReplicaOdometry.h"
#include "utils/Config.h"
#include "utils/Iterable.h"
#include "utils/Timers.h"
#include "slimvdb/Config.h"

using namespace fmt::literals;
using namespace utils;
namespace fs = std::filesystem;

namespace {

argparse::ArgumentParser ArgParse(int argc, char* argv[]) {
    argparse::ArgumentParser argparser("ReplicaPipeline");
    argparser.add_argument("replica_root_dir").help("Path to replica_niceslam/ root");
    argparser.add_argument("output_dir").help("Directory to write VDB + PLY outputs");
    argparser.add_argument("--sequence").help("Scene name (e.g. room0, office2)").required();
    argparser.add_argument("--config")
        .help("Dataset-specific config file")
        .default_value<std::string>("config/replica.yaml")
        .action([](const std::string& v) { return v; });
    argparser.add_argument("--n_scans")
        .help("Number of frames to integrate (-1 = all)")
        .default_value(int(-1))
        .action([](const std::string& v) { return std::stoi(v); });

    try {
        argparser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << "Invalid arguments: " << err.what() << std::endl;
        std::cerr << argparser;
        std::exit(1);
    }

    const auto root = argparser.get<std::string>("replica_root_dir");
    if (!fs::exists(root)) {
        std::cerr << "Replica root does not exist: " << root << std::endl;
        std::exit(1);
    }
    fs::create_directories(argparser.get<std::string>("output_dir"));
    return argparser;
}

struct RGB { int r, g, b; };

}  // namespace

int main(int argc, char* argv[]) {
    std::ofstream nullstream;
    std::clog.rdbuf(nullstream.rdbuf());

    auto argparser = ArgParse(argc, argv);

    const auto slimvdb_cfg = slimvdb::SLIMVDBConfig::LoadFromYAML(argparser.get<std::string>("--config"));
    const auto replica_cfg = datasets::ReplicaConfig::LoadFromYAML(argparser.get<std::string>("--config"));

    openvdb::initialize();

    const int n_scans = argparser.get<int>("--n_scans");
    const auto replica_root = argparser.get<std::string>("replica_root_dir");
    const auto sequence     = argparser.get<std::string>("--sequence");

    const auto dataset =
        datasets::ReplicaDataset<slimvdb::LANGUAGE>(replica_root, sequence, n_scans,
                                                    replica_cfg.apply_pose_,
                                                    replica_cfg.preprocess_,
                                                    replica_cfg.min_range_,
                                                    replica_cfg.max_range_,
                                                    replica_cfg.semantic_subdir_);

    fmt::print("Replica: integrating {} scans from scene '{}' (labels dir: {})\n",
               dataset.size(), sequence, replica_cfg.semantic_subdir_);

    slimvdb::VDBVolume<slimvdb::LANGUAGE> tsdf_volume(slimvdb_cfg.voxel_size_,
                                                     slimvdb_cfg.sdf_trunc_,
                                                     slimvdb_cfg.space_carving_,
                                                     slimvdb_cfg.min_weight_);
    timers::FPSTimer<10> timer;

    int index = 0;
    for (const auto& [scan, semantics, pose] : iterable(dataset)) {
        auto t1_s = std::chrono::high_resolution_clock::now();
        timer.tic();

        const Eigen::Vector3d origin = pose.block<3, 1>(0, 3);
        tsdf_volume.Integrate(scan, semantics, origin, [](float /*unused*/) { return 1.0; });

        auto t1_e = std::chrono::high_resolution_clock::now();

        // RENDER — optional; render resolution is driven by config.
        auto t2_s = std::chrono::high_resolution_clock::now();
        std::vector<double> origin_vec = {origin(0), origin(1), origin(2)};
        Eigen::Matrix3d rot = pose.block<3, 3>(0, 0);
        Eigen::Quaterniond e_quat(rot);
        Eigen::Quaterniond cf_quat(0, 1, 0, 0);
        e_quat = e_quat * cf_quat;
        std::vector<double> rot_quat_vec = {e_quat.x(), e_quat.y(), e_quat.z(), e_quat.w()};
        tsdf_volume.Render(origin_vec, rot_quat_vec, index,
                           replica_cfg.render_img_width_, replica_cfg.render_img_height_,
                           replica_cfg.min_range_, replica_cfg.max_range_,
                           slimvdb_cfg.p_threshold_);
        ++index;
        timer.toc();
        auto t2_e = std::chrono::high_resolution_clock::now();

        auto t3_s = std::chrono::high_resolution_clock::now();
        if (slimvdb_cfg.prune_interval_ >= 0 && index % slimvdb_cfg.prune_interval_ == 0) {
            tsdf_volume.tsdf_ = tsdf_volume.Prune(slimvdb_cfg.min_weight_);
        }
        auto t3_e = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> e1 = t1_e - t1_s;
        std::chrono::duration<double, std::milli> e2 = t2_e - t2_s;
        std::chrono::duration<double, std::milli> e3 = t3_e - t3_s;
        // Per-frame timing + map memory (machine-parseable) — downstream
        // scripts grep these lines for FPS distribution, P95, peak mem, etc.
        const uint64_t tsdf_bytes = tsdf_volume.tsdf_ ? tsdf_volume.tsdf_->memUsage() : 0;
        const uint64_t sem_bytes  = tsdf_volume.semantics_ ? tsdf_volume.semantics_->memUsage() : 0;
        fmt::print("TIMING idx={} integrate_ms={:.3f} render_ms={:.3f} prune_ms={:.3f} "
                   "vdb_tsdf_mb={:.2f} vdb_sem_mb={:.2f}\n",
                   index, e1.count(), e2.count(), e3.count(),
                   tsdf_bytes / (1024.0 * 1024.0), sem_bytes / (1024.0 * 1024.0));
        std::cout.flush();
    }

    // Save grids + a pointcloud for downstream evaluation.
    const std::string map_name = fmt::format("{out}/replica_{seq}_{n}",
                                             "out"_a = argparser.get<std::string>("output_dir"),
                                             "seq"_a = sequence,
                                             "n"_a = n_scans);
    {
        timers::ScopeTimer t("Writing TSDF VDB to disk");
        openvdb::io::File(map_name + "_tsdf.vdb").write({tsdf_volume.tsdf_});
    }
    {
        timers::ScopeTimer t("Writing semantic VDB to disk");
        openvdb::io::File(map_name + "_semantics.vdb").write({tsdf_volume.semantics_});
    }

    {
        timers::ScopeTimer t("Writing PointCloud (PLY) to disk");
        auto [points, labels] =
            tsdf_volume.ExtractPointCloud(false, slimvdb_cfg.min_weight_, slimvdb_cfg.p_threshold_);

        std::ofstream file(map_name + ".ply");
        if (!file.is_open()) throw std::runtime_error("could not open output .ply for writing");

        file << "ply\nformat ascii 1.0\n"
             << "element vertex " << points.size() << "\n"
             << "property float x\nproperty float y\nproperty float z\n"
             << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
             << "end_header\n";

        for (size_t i = 0; i < points.size(); ++i) {
            RGB color{255, 255, 255};
            auto it = replica_cfg.color_map_.find(static_cast<int>(labels[i]));
            if (it != replica_cfg.color_map_.end()) {
                color = {it->second[0], it->second[1], it->second[2]};
            }
            file << static_cast<float>(points[i].x()) << ' '
                 << static_cast<float>(points[i].y()) << ' '
                 << static_cast<float>(points[i].z()) << ' '
                 << color.r << ' ' << color.g << ' ' << color.b << '\n';
        }
    }

    return 0;
}
