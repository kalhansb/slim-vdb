// MIT License
//
// Reads a SLIM-VDB semantics `.vdb` grid (VecXI32<S> Dirichlet counts) and
// dumps each active voxel as a binary record: float32 wx, wy, wz, int32
// argmax_class. Optionally also writes a TSDF-occupancy mask filter.
//
// The output is a flat binary file — 16 bytes per record — consumed by the
// Python mIoU evaluation script. The compile-time NCLASSES is picked up from
// SLIM-VDB's Config.h via `#include <slimvdb/Config.h>`, so every variant
// (20 / 102 / 151 classes) produces a correctly-shaped tool binary.
//
// Usage:
//   vdb_to_voxels <semantics.vdb> <voxels.bin>

#include <openvdb/openvdb.h>
#include <slimvdb/Config.h>        // provides NCLASSES
#include <slimvdb/VDBVolume.h>     // provides openvdb::VecXI32 stub chain

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: vdb_to_voxels <semantics.vdb> <out.bin>\n";
        return 1;
    }
    const std::string vdb_path = argv[1];
    const std::string out_path = argv[2];

    openvdb::initialize();
    // VecXIGrid<N> isn't in OpenVDB's default GridTypes registry, so we have
    // to register the specific N we care about before trying to read a file
    // that contains it. NCLASSES is compile-time via slimvdb/Config.h.
    using LabelGridTypeOnly = openvdb::Grid<openvdb::tree::Tree4<openvdb::VecXI32<NCLASSES>, 5, 4, 3>::Type>;
    if (!LabelGridTypeOnly::isRegistered()) {
        LabelGridTypeOnly::registerGrid();
    }

    openvdb::io::File file(vdb_path);
    file.open();

    using LabelT = openvdb::VecXI32<NCLASSES>;
    using LabelGridT = openvdb::Grid<openvdb::tree::Tree4<LabelT, 5, 4, 3>::Type>;

    // Fetch the first grid. SLIM-VDB writes a single semantics grid per file.
    openvdb::GridBase::Ptr base_grid;
    for (auto it = file.beginName(); it != file.endName(); ++it) {
        std::cerr << "grid: name='" << it.gridName() << "'\n";
        base_grid = file.readGrid(it.gridName());
        std::cerr << "  loaded type=" << base_grid->type()
                  << " activeVoxels=" << base_grid->activeVoxelCount()
                  << " valueType=" << base_grid->valueType()
                  << "\n";
        break;
    }
    file.close();
    if (!base_grid) {
        std::cerr << "ERROR: no grid found in " << vdb_path << std::endl;
        return 2;
    }

    auto label_grid = openvdb::gridPtrCast<LabelGridT>(base_grid);
    if (!label_grid) {
        std::cerr << "ERROR: grid has unexpected type (expected VecXI32<"
                  << NCLASSES << ">) in " << vdb_path << std::endl;
        return 3;
    }

    const auto xform = label_grid->transform();

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: cannot open output " << out_path << std::endl;
        return 4;
    }

    uint64_t n_voxels = 0;
    uint64_t n_scanned = 0;
    uint64_t n_skipped_zero = 0;
    for (auto it = label_grid->cbeginValueOn(); it; ++it) {
        ++n_scanned;
        const openvdb::Coord c = it.getCoord();
        const openvdb::Vec3d w = xform.indexToWorld(c);
        const LabelT& counts = *it;

        // argmax over Dirichlet counts. VecXI32<N> stores unsigned Index32,
        // so we initialise best_val to 0 and treat any voxel whose max count
        // is still 0 as "no evidence".
        int best = -1;
        uint32_t best_val = 0;
        for (int i = 0; i < NCLASSES; ++i) {
            const uint32_t v = static_cast<uint32_t>(counts[i]);
            if (v > best_val) {
                best_val = v;
                best = i;
            }
        }
        if (best < 0) { ++n_skipped_zero; continue; }  // all-zero counts
        // Optional 3-line debug for first voxels — comment out for prod runs.
        if (n_scanned <= 1) {
            std::cerr << "  voxel[1] coord=("
                      << c.x() << "," << c.y() << "," << c.z()
                      << ") counts[0..4]=";
            for (int i = 0; i < std::min(5, NCLASSES); ++i) std::cerr << counts[i] << " ";
            std::cerr << " argmax=" << best << " val=" << best_val << "\n";
        }

        const float wx = static_cast<float>(w.x());
        const float wy = static_cast<float>(w.y());
        const float wz = static_cast<float>(w.z());
        const int32_t cls = static_cast<int32_t>(best);
        out.write(reinterpret_cast<const char*>(&wx), sizeof(wx));
        out.write(reinterpret_cast<const char*>(&wy), sizeof(wy));
        out.write(reinterpret_cast<const char*>(&wz), sizeof(wz));
        out.write(reinterpret_cast<const char*>(&cls), sizeof(cls));
        ++n_voxels;
    }
    out.close();

    std::cerr << "wrote " << n_voxels << "/" << n_scanned
              << " voxels from " << vdb_path
              << " (skipped_zero_counts=" << n_skipped_zero
              << ") NCLASSES=" << NCLASSES << "\n";
    return 0;
}
