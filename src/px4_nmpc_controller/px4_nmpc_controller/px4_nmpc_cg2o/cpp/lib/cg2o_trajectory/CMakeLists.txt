add_library(cg2o_trajectory
  mpc_formulation.cpp
  linear_solvers.cpp
)

target_include_directories(cg2o_trajectory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cg2o_trajectory PUBLIC
  cg2o::core
  g2o::core
  g2o::stuff
  Eigen3::Eigen
)

# Optional linear solvers: compiled in when this g2o/cg2o installation exports them.
# umfpack is the cg2o default, qdldlt the fastest on indefinite KKT systems.
macro(_px4_nmpc_solver_feature target define)
  if(TARGET ${target})
    target_link_libraries(cg2o_trajectory PUBLIC ${target})
    target_compile_definitions(cg2o_trajectory PUBLIC ${define}=1)
    message(STATUS "cg2o backend linear solver: ${target}")
  endif()
endmacro()
_px4_nmpc_solver_feature(cg2o::solver_eigen_umfpack PX4_NMPC_HAVE_CG2O_UMFPACK)
_px4_nmpc_solver_feature(cg2o::solver_qdldlt PX4_NMPC_HAVE_CG2O_QDLDLT)
_px4_nmpc_solver_feature(cg2o::solvers_eigen PX4_NMPC_HAVE_CG2O_EIGEN)
_px4_nmpc_solver_feature(cg2o::solver_cholmod_fast PX4_NMPC_HAVE_CG2O_CHOLMOD_FAST)
_px4_nmpc_solver_feature(g2o::solver_cholmod PX4_NMPC_HAVE_G2O_CHOLMOD)
_px4_nmpc_solver_feature(g2o::solver_csparse PX4_NMPC_HAVE_G2O_CSPARSE)

target_compile_features(cg2o_trajectory PUBLIC cxx_std_20)
