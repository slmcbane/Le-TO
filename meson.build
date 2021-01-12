project('Le-TO', ['cpp'], default_options : ['cpp_std=c++17'])

executable('inspect-density-filter', 'inspect-density-filter.cpp',
           'density-filter.cpp', 'read_mesh.cpp')

executable('convert_mesh', 'convert_mesh.cpp', 'read_mesh.cpp', 'options.cpp')

executable('forward_solve_only', 'forward_solve_only.cpp', 'read_mesh.cpp',
           'options.cpp', 'forward_model.cpp', 'density-filter.cpp')
