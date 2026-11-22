pybind11_add_module(cpp_px4_nmpc_cg2o cg2o_trajectory_solver.cpp)

target_link_libraries(cpp_px4_nmpc_cg2o PRIVATE cg2o_trajectory)

# Land the extension at the build root so px4_nmpc_controller/px4_nmpc_cg2o/solver.py can
# find it with a plain PYTHONPATH entry, the same convention learn.zip's own
# build_pybind11.sh used for cpp_mpc_drone_solver.
set_target_properties(cpp_px4_nmpc_cg2o PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# DT_RPATH (not RUNPATH) so the indirect g2o dependencies (e.g. libg2o_csparse_extension) are found
# without sourcing setup_env.sh.
target_link_options(cpp_px4_nmpc_cg2o PRIVATE -Wl,--disable-new-dtags)
set_target_properties(cpp_px4_nmpc_cg2o PROPERTIES
  BUILD_RPATH "${PX4_NMPC_G2O_PREFIX}/lib;${PX4_NMPC_CG2O_PREFIX}/lib")
