# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_repo_root ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(_rsv_ut_include_dirs
    ${_repo_root}/experimental/ops/reduce_scatter_v/birs
    ${_repo_root}/experimental/ops/reduce_scatter_v/birs/template
    ${_repo_root}/experimental/ops/op_common/template
    ${_repo_root}/src/ops/op_common/inc
    ${_repo_root}/src/ops/op_common
    ${_repo_root}/src/ops/op_common/algorithm/template
    ${_repo_root}/src/ops/op_common/algorithm/template/registry
    ${_repo_root}/src/ops/op_common/algorithm/template/wrapper
    ${_repo_root}/src/ops/op_common/algorithm/executor
    ${_repo_root}/src/ops/op_common/algorithm/executor/registry
    ${_repo_root}/src/ops/op_common/algorithm/executor/channel
    ${_repo_root}/src/ops/op_common/algorithm/topo_match
    ${_repo_root}/src/ops/op_common/selector
    ${_repo_root}/src/ops/op_common/topo_info
    ${_repo_root}/src/common
    ${_repo_root}/src/common/hcomm_dlsym
    ${_repo_root}/src/common/hcomm_dlsym/ccu
)

# 前向兼容：探测 CANN 版本号，与主构建/ST 保持一致（宏逻辑与 test/ut/recursive_executor 一致）。
# 缺少该宏时 dlsym_common.h 中的桩类型块会启用，与 CANN 公共头重复定义导致 conflicting declaration。
macro(_detect_cann_version _target)
    file(GLOB _cann_arch_dirs LIST_DIRECTORIES true "${ASCEND_CANN_PACKAGE_PATH}/*-linux")
    if(_cann_arch_dirs)
        set(_cann_arch_dir "")
        foreach(_dir ${_cann_arch_dirs})
            if(_dir MATCHES "/${CMAKE_SYSTEM_PROCESSOR}-linux$")
                set(_cann_arch_dir ${_dir})
                break()
            endif()
        endforeach()
        if(NOT _cann_arch_dir)
            list(GET _cann_arch_dirs 0 _cann_arch_dir)
        endif()
        target_include_directories(${_target} PRIVATE
            ${_cann_arch_dir}/include
            ${_cann_arch_dir}/include/hccl
            ${_cann_arch_dir}/include/hcomm
            ${_cann_arch_dir}/include/hcomm/ccu
            ${_cann_arch_dir}/pkg_inc
            ${_cann_arch_dir}/pkg_inc/hccl
            ${_cann_arch_dir}/pkg_inc/base
            ${_cann_arch_dir}/pkg_inc/hcomm/ccu
        )
        foreach(_arch_dir ${_cann_arch_dirs})
            set(_ut_cann_ver_header "${_arch_dir}/include/version/cann_version.h")
            if(EXISTS "${_ut_cann_ver_header}")
                file(STRINGS "${_ut_cann_ver_header}" _ut_cann_ver_line
                     REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
                if(_ut_cann_ver_line)
                    string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
                        _UT_CANN_VERSION_EXPR "${_ut_cann_ver_line}")
                    math(EXPR _UT_CANN_VERSION_NUM "${_UT_CANN_VERSION_EXPR}" OUTPUT_FORMAT DECIMAL)
                    target_compile_definitions(${_target}
                        PRIVATE CANN_VERSION_NUM=${_UT_CANN_VERSION_NUM})
                    message(STATUS "[UT] Detected CANN_VERSION_NUM = ${_UT_CANN_VERSION_NUM}")
                else()
                    message(FATAL_ERROR "[UT] Failed to parse CANN_VERSION_NUM from ${_ut_cann_ver_header}: "
                        "no '#define CANN_VERSION_NUM' line found. "
                        "dlsym_common.h stub blocks would collide with CANN public headers.")
                endif()
                break()
            endif()
        endforeach()
    endif()
endmacro()

# ---------- reduce_scatter_v_birs_selector_test: 选路判定纯逻辑 ----------
set(ut_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/birs_v_selector_test.cc
    ${_repo_root}/experimental/ops/reduce_scatter_v/birs/reduce_scatter_v_birs_selector.cc
)

add_executable(reduce_scatter_v_birs_selector_test ${ut_src_list})

target_compile_options(reduce_scatter_v_birs_selector_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(reduce_scatter_v_birs_selector_test PRIVATE ${_rsv_ut_include_dirs})

_detect_cann_version(reduce_scatter_v_birs_selector_test)

target_link_libraries(reduce_scatter_v_birs_selector_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET reduce_scatter_v_birs_selector_test TASK_NUM 1)

# ---------- reduce_scatter_v_birs_core_test: 变长核心纯计算逻辑 ----------
# executor_base/template 源文件整体编译; HCOMM 原语与 AlgTemplateBase 运行时依赖由本目录桩提供, 不依赖硬件
set(core_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/birs_v_core_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/base_class_stub.cc
    ${_repo_root}/experimental/ops/reduce_scatter_v/birs/reduce_scatter_v_executor_base.cc
    ${_repo_root}/experimental/ops/reduce_scatter_v/birs/template/reduce_scatter_v_birs.cc
    ${_repo_root}/experimental/ops/reduce_scatter_v/birs/template/reduce_scatter_v_birs_inter.cc
    ${_repo_root}/experimental/ops/op_common/template/alg_template_base_experimental.cc
    ${_repo_root}/src/ops/op_common/algorithm/executor/executor_base.cc
    ${_repo_root}/src/ops/op_common/algorithm/template/registry/alg_template_register.cc
)

add_executable(reduce_scatter_v_birs_core_test ${core_src_list})

target_compile_options(reduce_scatter_v_birs_core_test PRIVATE
    -U_FORTIFY_SOURCE
)

target_include_directories(reduce_scatter_v_birs_core_test PRIVATE ${_rsv_ut_include_dirs})

_detect_cann_version(reduce_scatter_v_birs_core_test)

target_link_libraries(reduce_scatter_v_birs_core_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
    -lpthread
)

run_llt_test(TARGET reduce_scatter_v_birs_core_test TASK_NUM 1)
