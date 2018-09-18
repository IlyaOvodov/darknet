#!/usr/bin/cmake -P
#This file is expected to be executed in "cmake script mode" - via cmake -P
cmake_minimum_required(VERSION 3.2)
get_filename_component(THIRD_PARTY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE CACHE)
file(TO_CMAKE_PATH "${THIRD_PARTY_ROOT}" THIRD_PARTY_ROOT)
list(APPEND CMAKE_MODULE_PATH "${THIRD_PARTY_ROOT}/build_helpers/cmake_common_v001")
include(third_party_paths)
include(third_party_build_utils)
set(extra_opts_Debug "-O0")
set(extra_opts_Release "-Ofast")
set(suffix_use_gpu_0 "_no_gpu")
set(only_supported_nvcc_path "/usr/local/cuda/bin/nvcc")# Makefile already has hard-coded path to cuda includes
foreach(config "Release" "Debug")
	create_third_party_variant_install(variant_install "${CMAKE_CURRENT_LIST_DIR}" "${config}" "gcc")
	file(MAKE_DIRECTORY "${variant_install}/bin")
	file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/build/darknet/YoloDetector.h" DESTINATION "${variant_install}/include/darknet")
	foreach(use_gpu "1" "0")
		if(${use_gpu} EQUAL "0" OR EXISTS "${only_supported_nvcc_path}") # skip gpu build if nvcc is not found
			set(make_dir "${THIRD_PARTY_TMP}/${config}-use_gpu_${use_gpu}")
			set(lib_so "libyolo_cpp_dll${suffix_use_gpu_${use_gpu}}.so")
			file(COPY "${CMAKE_CURRENT_LIST_DIR}/" DESTINATION "${make_dir}" PATTERN ".svn" EXCLUDE PATTERN ".git" EXCLUDE)

			execute_process(COMMAND "make"
			"OPTS=-g ${extra_opts_${config}}"
			"LIBSO=1"
			"LIBNAMESO=${lib_so}"
			"EXEC=${variant_install}/bin/darknet${suffix_use_gpu_${use_gpu}}"
			"APPNAMESO=${variant_install}/bin/darknet_uselib${suffix_use_gpu_${use_gpu}}"
			"GPU=${use_gpu}"
			"CUDNN=${use_gpu}"
			"NVCC=${only_supported_nvcc_path}"
			WORKING_DIRECTORY "${make_dir}"
			RESULT_VARIABLE make_status)
			if (NOT ${make_status} EQUAL 0)
				message(SEND_ERROR "make failed with code '${make_status}'")
			endif()
			file(INSTALL "${make_dir}/${lib_so}" DESTINATION "${variant_install}/lib/")
		endif()
	endforeach()
endforeach()
