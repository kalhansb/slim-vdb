#include "common.h"
#include "nanovdb.cuh"


template void runNanoVDB<slimvdb::CLOSED, NCLASSES>(
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    int, int,
    nanovdb::cuda::DeviceBuffer&,
    int,
    std::vector<double>,
    std::vector<double>,
    const float,
    const float,
    const float, 
    const float*,
    const uint16_t
);

// Local patch (2026-04-14): OPEN-set instantiation disabled because the
// umfieldrobotics/openvdb slim-vdb branch does not expose `nanovdb::math::VecXf<S>`
// or `nanovdb::VecXFGrid<S>`, which this template instantiation requires. Our
// experiments only use closed-set semantics (KITTI NCLASSES=20, Replica
// NCLASSES=102), so leaving only the CLOSED instantiation is sufficient.
//
// If you need open-set again, add VecXf/VecXFGrid to the openvdb fork and
// re-enable this block.
#if 0
template void runNanoVDB<slimvdb::OPEN, NCLASSES>(
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&,
    int, int,
    nanovdb::cuda::DeviceBuffer&,
    int,
    std::vector<double>,
    std::vector<double>,
    const float,
    const float,
    const float,
    const float*,
    const uint16_t
);
#endif