# Written by Anja Sheppard, February 2024
# Local patch (2026-04-14): hardcoded HINTS path removed.
# Pass -DLIBTORCH_PREFIX=/path/to/libtorch (or include it in CMAKE_PREFIX_PATH).

if(DEFINED LIBTORCH_PREFIX)
    find_package(Torch REQUIRED HINTS "${LIBTORCH_PREFIX}")
else()
    find_package(Torch REQUIRED)
endif()