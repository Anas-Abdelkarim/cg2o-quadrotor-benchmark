pybind11_add_module(cpp_px4_nmpc_ipopt_cpp ipopt_trajectory_pybind.cpp)
target_link_libraries(cpp_px4_nmpc_ipopt_cpp PRIVATE px4_nmpc_ipopt_cpp_core)
target_include_directories(cpp_px4_nmpc_ipopt_cpp PRIVATE ../lib/ipopt_trajectory)
target_compile_features(cpp_px4_nmpc_ipopt_cpp PRIVATE cxx_std_17)

# Land the extension at the build root (build/ipopt_cpp_backend/) so that
# px4_nmpc_ipopt_cpp/solver.py finds it, exactly like the cg2o extension.
set_target_properties(cpp_px4_nmpc_ipopt_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# DT_RPATH to the project-local IPOPT (and its MUMPS/SPRAL libraries): no LD_LIBRARY_PATH needed
if(PX4_NMPC_IPOPT_RPATH)
  target_link_options(cpp_px4_nmpc_ipopt_cpp PRIVATE -Wl,--disable-new-dtags)
  set_target_properties(cpp_px4_nmpc_ipopt_cpp PROPERTIES BUILD_RPATH "${PX4_NMPC_IPOPT_RPATH}")
endif()
