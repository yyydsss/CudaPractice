# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yds

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yds/build

# Include any dependencies generated for this target.
include 4_gemm/CMakeFiles/gemm_0_global.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 4_gemm/CMakeFiles/gemm_0_global.dir/compiler_depend.make

# Include the progress variables for this target.
include 4_gemm/CMakeFiles/gemm_0_global.dir/progress.make

# Include the compile flags for this target's objects.
include 4_gemm/CMakeFiles/gemm_0_global.dir/flags.make

4_gemm/CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.o: 4_gemm/CMakeFiles/gemm_0_global.dir/flags.make
4_gemm/CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.o: ../4_gemm/gemm_0_global.cu
4_gemm/CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.o: 4_gemm/CMakeFiles/gemm_0_global.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yds/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 4_gemm/CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.o"
	cd /home/yds/build/4_gemm && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT 4_gemm/CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.o -MF CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.o.d -x cu -c /home/yds/4_gemm/gemm_0_global.cu -o CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.o

4_gemm/CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

4_gemm/CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target gemm_0_global
gemm_0_global_OBJECTS = \
"CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.o"

# External object files for target gemm_0_global
gemm_0_global_EXTERNAL_OBJECTS =

4_gemm/gemm_0_global: 4_gemm/CMakeFiles/gemm_0_global.dir/gemm_0_global.cu.o
4_gemm/gemm_0_global: 4_gemm/CMakeFiles/gemm_0_global.dir/build.make
4_gemm/gemm_0_global: /usr/local/cuda/lib64/libcudart.so
4_gemm/gemm_0_global: /usr/local/cuda/lib64/libcublas.so
4_gemm/gemm_0_global: 4_gemm/CMakeFiles/gemm_0_global.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yds/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable gemm_0_global"
	cd /home/yds/build/4_gemm && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gemm_0_global.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
4_gemm/CMakeFiles/gemm_0_global.dir/build: 4_gemm/gemm_0_global
.PHONY : 4_gemm/CMakeFiles/gemm_0_global.dir/build

4_gemm/CMakeFiles/gemm_0_global.dir/clean:
	cd /home/yds/build/4_gemm && $(CMAKE_COMMAND) -P CMakeFiles/gemm_0_global.dir/cmake_clean.cmake
.PHONY : 4_gemm/CMakeFiles/gemm_0_global.dir/clean

4_gemm/CMakeFiles/gemm_0_global.dir/depend:
	cd /home/yds/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yds /home/yds/4_gemm /home/yds/build /home/yds/build/4_gemm /home/yds/build/4_gemm/CMakeFiles/gemm_0_global.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 4_gemm/CMakeFiles/gemm_0_global.dir/depend

