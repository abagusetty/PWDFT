
# files in the nwpwlib library
file(GLOB_RECURSE src_blas      blas/*.f)
file(GLOB_RECURSE src_lapack    lapack/*.f)
file(GLOB_RECURSE src_fftpack   fftpack/*.f)
file(GLOB_RECURSE src_Control   Control/*.hpp Control/*.cpp Control/*.h Control/*.c)
file(GLOB_RECURSE src_Parallel  Parallel/*.hpp Parallel/*.cpp)
file(GLOB_RECURSE src_io        io/*.hpp io/*.cpp)
file(GLOB_RECURSE src_ion       ion/*.hpp ion/*.cpp)
file(GLOB_RECURSE src_utilities utilities/*.hpp utilities/*.cpp)
file(GLOB_RECURSE src_lattice   lattice/*.hpp lattice/*.cpp)
file(GLOB_RECURSE src_D3dB      D3dB/*.hpp D3dB/*.cpp)
file(GLOB_RECURSE src_pspgen    pseudopotential/*.h  pseudopotential/*.c)
file(GLOB_RECURSE src_psplib    psp_library/*.hpp  psp_library/*.cpp)
file(GLOB_RECURSE src_parse     parse/*.hpp parse/*.cpp)


# create the nwpwlib library
add_library(nwpwlib ${src_D3dB} ${src_ion} ${src_lattice} ${src_Control} ${src_io} ${src_utilities} ${src_Parallel} ${src_fftpack} ${src_pspgen} ${src_psplib} ${src_parse})

# adding target_include_directories to nwpwlib
add_subdirectory(parse)
add_subdirectory(psp_library)
add_subdirectory(pseudopotential)
add_subdirectory(pseudopotential/paw_atom)
#add_subdirectory(blas)
add_subdirectory(fftpack)
add_subdirectory(Control)
add_subdirectory(Parallel)
add_subdirectory(utilities)
add_subdirectory(io)
add_subdirectory(lattice)
add_subdirectory(ion)
add_subdirectory(D3dB)

