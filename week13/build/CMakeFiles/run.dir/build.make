# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.30.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.30.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/giyoung/Desktop/수업/4학기/computerVision/week13

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/giyoung/Desktop/수업/4학기/computerVision/week13/build

# Include any dependencies generated for this target.
include CMakeFiles/run.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/run.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/run.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/run.dir/flags.make

CMakeFiles/run.dir/22000053.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/22000053.cpp.o: /Users/giyoung/Desktop/수업/4학기/computerVision/week13/22000053.cpp
CMakeFiles/run.dir/22000053.cpp.o: CMakeFiles/run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/giyoung/Desktop/수업/4학기/computerVision/week13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/run.dir/22000053.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run.dir/22000053.cpp.o -MF CMakeFiles/run.dir/22000053.cpp.o.d -o CMakeFiles/run.dir/22000053.cpp.o -c /Users/giyoung/Desktop/수업/4학기/computerVision/week13/22000053.cpp

CMakeFiles/run.dir/22000053.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run.dir/22000053.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/giyoung/Desktop/수업/4학기/computerVision/week13/22000053.cpp > CMakeFiles/run.dir/22000053.cpp.i

CMakeFiles/run.dir/22000053.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run.dir/22000053.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/giyoung/Desktop/수업/4학기/computerVision/week13/22000053.cpp -o CMakeFiles/run.dir/22000053.cpp.s

# Object files for target run
run_OBJECTS = \
"CMakeFiles/run.dir/22000053.cpp.o"

# External object files for target run
run_EXTERNAL_OBJECTS =

run: CMakeFiles/run.dir/22000053.cpp.o
run: CMakeFiles/run.dir/build.make
run: /usr/local/lib/libopencv_gapi.4.10.0.dylib
run: /usr/local/lib/libopencv_stitching.4.10.0.dylib
run: /usr/local/lib/libopencv_alphamat.4.10.0.dylib
run: /usr/local/lib/libopencv_aruco.4.10.0.dylib
run: /usr/local/lib/libopencv_bgsegm.4.10.0.dylib
run: /usr/local/lib/libopencv_bioinspired.4.10.0.dylib
run: /usr/local/lib/libopencv_ccalib.4.10.0.dylib
run: /usr/local/lib/libopencv_dnn_objdetect.4.10.0.dylib
run: /usr/local/lib/libopencv_dnn_superres.4.10.0.dylib
run: /usr/local/lib/libopencv_dpm.4.10.0.dylib
run: /usr/local/lib/libopencv_face.4.10.0.dylib
run: /usr/local/lib/libopencv_freetype.4.10.0.dylib
run: /usr/local/lib/libopencv_fuzzy.4.10.0.dylib
run: /usr/local/lib/libopencv_hfs.4.10.0.dylib
run: /usr/local/lib/libopencv_img_hash.4.10.0.dylib
run: /usr/local/lib/libopencv_intensity_transform.4.10.0.dylib
run: /usr/local/lib/libopencv_line_descriptor.4.10.0.dylib
run: /usr/local/lib/libopencv_mcc.4.10.0.dylib
run: /usr/local/lib/libopencv_quality.4.10.0.dylib
run: /usr/local/lib/libopencv_rapid.4.10.0.dylib
run: /usr/local/lib/libopencv_reg.4.10.0.dylib
run: /usr/local/lib/libopencv_rgbd.4.10.0.dylib
run: /usr/local/lib/libopencv_saliency.4.10.0.dylib
run: /usr/local/lib/libopencv_sfm.4.10.0.dylib
run: /usr/local/lib/libopencv_signal.4.10.0.dylib
run: /usr/local/lib/libopencv_stereo.4.10.0.dylib
run: /usr/local/lib/libopencv_structured_light.4.10.0.dylib
run: /usr/local/lib/libopencv_superres.4.10.0.dylib
run: /usr/local/lib/libopencv_surface_matching.4.10.0.dylib
run: /usr/local/lib/libopencv_tracking.4.10.0.dylib
run: /usr/local/lib/libopencv_videostab.4.10.0.dylib
run: /usr/local/lib/libopencv_viz.4.10.0.dylib
run: /usr/local/lib/libopencv_wechat_qrcode.4.10.0.dylib
run: /usr/local/lib/libopencv_xfeatures2d.4.10.0.dylib
run: /usr/local/lib/libopencv_xobjdetect.4.10.0.dylib
run: /usr/local/lib/libopencv_xphoto.4.10.0.dylib
run: /usr/local/lib/libopencv_shape.4.10.0.dylib
run: /usr/local/lib/libopencv_highgui.4.10.0.dylib
run: /usr/local/lib/libopencv_datasets.4.10.0.dylib
run: /usr/local/lib/libopencv_plot.4.10.0.dylib
run: /usr/local/lib/libopencv_text.4.10.0.dylib
run: /usr/local/lib/libopencv_ml.4.10.0.dylib
run: /usr/local/lib/libopencv_phase_unwrapping.4.10.0.dylib
run: /usr/local/lib/libopencv_optflow.4.10.0.dylib
run: /usr/local/lib/libopencv_ximgproc.4.10.0.dylib
run: /usr/local/lib/libopencv_video.4.10.0.dylib
run: /usr/local/lib/libopencv_videoio.4.10.0.dylib
run: /usr/local/lib/libopencv_imgcodecs.4.10.0.dylib
run: /usr/local/lib/libopencv_objdetect.4.10.0.dylib
run: /usr/local/lib/libopencv_calib3d.4.10.0.dylib
run: /usr/local/lib/libopencv_dnn.4.10.0.dylib
run: /usr/local/lib/libopencv_features2d.4.10.0.dylib
run: /usr/local/lib/libopencv_flann.4.10.0.dylib
run: /usr/local/lib/libopencv_photo.4.10.0.dylib
run: /usr/local/lib/libopencv_imgproc.4.10.0.dylib
run: /usr/local/lib/libopencv_core.4.10.0.dylib
run: CMakeFiles/run.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/giyoung/Desktop/수업/4학기/computerVision/week13/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable run"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/run.dir/build: run
.PHONY : CMakeFiles/run.dir/build

CMakeFiles/run.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run.dir/clean

CMakeFiles/run.dir/depend:
	cd /Users/giyoung/Desktop/수업/4학기/computerVision/week13/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/giyoung/Desktop/수업/4학기/computerVision/week13 /Users/giyoung/Desktop/수업/4학기/computerVision/week13 /Users/giyoung/Desktop/수업/4학기/computerVision/week13/build /Users/giyoung/Desktop/수업/4학기/computerVision/week13/build /Users/giyoung/Desktop/수업/4학기/computerVision/week13/build/CMakeFiles/run.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/run.dir/depend

