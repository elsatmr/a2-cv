# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.27.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.27.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/elsatamara/Downloads/csv_util

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/elsatamara/Downloads/csv_util/bin

# Include any dependencies generated for this target.
include CMakeFiles/FeatureVectorWriter.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/FeatureVectorWriter.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/FeatureVectorWriter.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FeatureVectorWriter.dir/flags.make

CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.o: CMakeFiles/FeatureVectorWriter.dir/flags.make
CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.o: /Users/elsatamara/Downloads/csv_util/src/featureVectorWriter.cpp
CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.o: CMakeFiles/FeatureVectorWriter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/elsatamara/Downloads/csv_util/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.o -MF CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.o.d -o CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.o -c /Users/elsatamara/Downloads/csv_util/src/featureVectorWriter.cpp

CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/elsatamara/Downloads/csv_util/src/featureVectorWriter.cpp > CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.i

CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/elsatamara/Downloads/csv_util/src/featureVectorWriter.cpp -o CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.s

CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.o: CMakeFiles/FeatureVectorWriter.dir/flags.make
CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.o: /Users/elsatamara/Downloads/csv_util/src/helpers.cpp
CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.o: CMakeFiles/FeatureVectorWriter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/elsatamara/Downloads/csv_util/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.o -MF CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.o.d -o CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.o -c /Users/elsatamara/Downloads/csv_util/src/helpers.cpp

CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/elsatamara/Downloads/csv_util/src/helpers.cpp > CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.i

CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/elsatamara/Downloads/csv_util/src/helpers.cpp -o CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.s

CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.o: CMakeFiles/FeatureVectorWriter.dir/flags.make
CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.o: /Users/elsatamara/Downloads/csv_util/src/csv_util.cpp
CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.o: CMakeFiles/FeatureVectorWriter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/elsatamara/Downloads/csv_util/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.o -MF CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.o.d -o CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.o -c /Users/elsatamara/Downloads/csv_util/src/csv_util.cpp

CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/elsatamara/Downloads/csv_util/src/csv_util.cpp > CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.i

CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/elsatamara/Downloads/csv_util/src/csv_util.cpp -o CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.s

# Object files for target FeatureVectorWriter
FeatureVectorWriter_OBJECTS = \
"CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.o" \
"CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.o" \
"CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.o"

# External object files for target FeatureVectorWriter
FeatureVectorWriter_EXTERNAL_OBJECTS =

FeatureVectorWriter: CMakeFiles/FeatureVectorWriter.dir/src/featureVectorWriter.cpp.o
FeatureVectorWriter: CMakeFiles/FeatureVectorWriter.dir/src/helpers.cpp.o
FeatureVectorWriter: CMakeFiles/FeatureVectorWriter.dir/src/csv_util.cpp.o
FeatureVectorWriter: CMakeFiles/FeatureVectorWriter.dir/build.make
FeatureVectorWriter: /usr/local/lib/libopencv_gapi.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_stitching.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_alphamat.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_aruco.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_bgsegm.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_bioinspired.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_ccalib.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_dnn_objdetect.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_dnn_superres.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_dpm.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_face.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_freetype.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_fuzzy.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_hfs.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_img_hash.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_intensity_transform.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_line_descriptor.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_mcc.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_quality.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_rapid.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_reg.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_rgbd.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_saliency.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_sfm.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_stereo.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_structured_light.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_superres.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_surface_matching.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_tracking.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_videostab.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_viz.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_wechat_qrcode.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_xfeatures2d.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_xobjdetect.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_xphoto.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_shape.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_highgui.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_datasets.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_plot.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_text.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_ml.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_phase_unwrapping.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_optflow.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_ximgproc.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_video.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_videoio.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_imgcodecs.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_objdetect.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_calib3d.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_dnn.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_features2d.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_flann.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_photo.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_imgproc.4.8.0.dylib
FeatureVectorWriter: /usr/local/lib/libopencv_core.4.8.0.dylib
FeatureVectorWriter: CMakeFiles/FeatureVectorWriter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/elsatamara/Downloads/csv_util/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable FeatureVectorWriter"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FeatureVectorWriter.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FeatureVectorWriter.dir/build: FeatureVectorWriter
.PHONY : CMakeFiles/FeatureVectorWriter.dir/build

CMakeFiles/FeatureVectorWriter.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FeatureVectorWriter.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FeatureVectorWriter.dir/clean

CMakeFiles/FeatureVectorWriter.dir/depend:
	cd /Users/elsatamara/Downloads/csv_util/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/elsatamara/Downloads/csv_util /Users/elsatamara/Downloads/csv_util /Users/elsatamara/Downloads/csv_util/bin /Users/elsatamara/Downloads/csv_util/bin /Users/elsatamara/Downloads/csv_util/bin/CMakeFiles/FeatureVectorWriter.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/FeatureVectorWriter.dir/depend
