# zlib
set(zlib_version 1.2.11)
set(zlib_url "https://github.com/madler/zlib/archive/v${zlib_version}.zip")
set(zlib_md5 "9d6a627693163bbbf3f26403a3a0b0b1")
set(zlib_dlname "zlib-${zlib_version}.zip")
list(APPEND ALL_PROJECTS zlib)

# lapack (blas)
if(NOT APPLE)
  set(lapack_version 3.10.0)
  set(lapack_url "https://github.com/Reference-LAPACK/lapack/archive/v${lapack_version}.tar.gz")
  set(lapack_md5 "d70fc27a8bdebe00481c97c728184f09")
  list(APPEND ALL_PROJECTS lapack)
endif()
