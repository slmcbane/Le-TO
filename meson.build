project('Le-TO', ['cpp'], default_options : ['cpp_std=c++17'])

cppc = meson.get_compiler('cpp')

ipopt = cppc.find_library('ipopt', dirs : get_option('ipopt_dir') + '/lib',
                          header_args : '-I' + get_option('ipopt_dir') + '/include/coin-or',
                          has_headers : ['IpTNLP.hpp'])

blas = dependency('openblas')
#blas = cppc.find_library(get_option('blaslib'))

incdirlist = include_directories([get_option('ipopt_dir') + '/include/coin-or'], get_option('eigen_include_dir'))

leto_lib = static_library('leto_lib', 'density-filter.cpp', 'read_mesh.cpp', 'options.cpp',
                          'forward_model.cpp', 'density-filter.cpp', 'evaluator.cpp', 'stress.cpp',
                          include_directories : incdirlist)

executable('inspect-density-filter', 'inspect-density-filter.cpp',
           link_with : leto_lib, include_directories : incdirlist)

executable('convert_mesh', 'convert_mesh.cpp', link_with : leto_lib,
           include_directories : incdirlist)

executable('forward_solve_only', 'forward_solve_only.cpp', link_with : leto_lib,
           include_directories : incdirlist)

executable('compliance_gradient_check', 'compliance_gradient_check.cpp', link_with : leto_lib,
           include_directories : incdirlist)

executable('stress_jacobian_check', 'stress_jacobian_check.cpp', link_with : leto_lib,
           include_directories : incdirlist)

executable('run_optimization', 'run_optimization.cpp', 'OptimizationProblem.cpp',
           link_with : leto_lib, dependencies : [ipopt, blas],
           include_directories : incdirlist)

executable('postprocess', 'postprocess.cpp', link_with : leto_lib,
           include_directories : incdirlist)