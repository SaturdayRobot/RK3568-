set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(RK3568_SDK_ROOT "/home/saturday/rk3568_linux_sdk" CACHE PATH "RK3568 Linux SDK root")
set(CMAKE_SYSROOT "${RK3568_SDK_ROOT}/debian/binary")

set(RK3568_TOOLCHAIN_ROOT
    "${RK3568_SDK_ROOT}/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu"
    CACHE PATH "RK3568 GCC toolchain root"
)
set(CMAKE_CXX_COMPILER "${RK3568_TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++")

# The SDK compiler's bundled libstdc++ targets glibc 2.33, while the board
# rootfs is Debian 11/glibc 2.31. Force Debian's CRT and libstdc++ so the
# resulting executable cannot accidentally acquire a newer GLIBC dependency.
set(_DEBIAN_LINK_DIR "${CMAKE_BINARY_DIR}/debian11-link-libs")
set(_DEBIAN_INCLUDE_DIR "${CMAKE_BINARY_DIR}/debian11-override-include")
file(MAKE_DIRECTORY "${_DEBIAN_LINK_DIR}")
file(MAKE_DIRECTORY "${_DEBIAN_INCLUDE_DIR}")
file(CREATE_LINK
    "${CMAKE_SYSROOT}/usr/include/pthread.h"
    "${_DEBIAN_INCLUDE_DIR}/pthread.h"
    SYMBOLIC
)
file(CREATE_LINK
    "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/libstdc++.so.6"
    "${_DEBIAN_LINK_DIR}/libstdc++.so"
    SYMBOLIC
)
# The extracted rootfs keeps several development symlinks as absolute target
# paths (for example /lib/aarch64-linux-gnu/libpthread.so.0).  Those links are
# valid on the board but not from the host linker, so expose host-resolvable
# links in a staging directory searched before the sysroot archives.
set(_DEBIAN_RUNTIME_LINKS pthread.so.0 dl.so.2 m.so.6 rt.so.1)
foreach(_LIB_LINK IN LISTS _DEBIAN_RUNTIME_LINKS)
    string(REGEX REPLACE "\\.so\\..*$" "" _LIB_NAME "${_LIB_LINK}")
    if(EXISTS "${CMAKE_SYSROOT}/lib/aarch64-linux-gnu/lib${_LIB_LINK}")
        file(CREATE_LINK
            "${CMAKE_SYSROOT}/lib/aarch64-linux-gnu/lib${_LIB_LINK}"
            "${_DEBIAN_LINK_DIR}/lib${_LIB_NAME}.so"
            SYMBOLIC
        )
    endif()
endforeach()
set(CMAKE_CXX_FLAGS_INIT
    "-B${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/ -isystem${_DEBIAN_INCLUDE_DIR} -isystem${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu -L${_DEBIAN_LINK_DIR} -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu -L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu"
)

set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu -Wl,-rpath-link,${CMAKE_SYSROOT}/lib/aarch64-linux-gnu -Wl,--allow-shlib-undefined"
)
