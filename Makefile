# Minimal J2K-only static library build for OpenJPEG sources (C++ Version).
# - Builds the JPEG-2000 codestream (J2K) core from src/lib/openjp2
# - Builds a tiny PNM (P5/P6) -> .j2k encoder executable.
# - Does NOT build any upstream applications (src/bin/*), JPIP, wrappers, etc.

OPENJP2_DIR := src
COMMON_DIR  := src/common
BUILD_DIR   := build

CXX      ?= g++
MPIXX    ?= mpicxx
AR       ?= ar
RANLIB   ?= ranlib

# Optimization flags
CXXFLAGS  ?= -O3 -march=native -mtune=native
CXXFLAGS  += -ffast-math -funroll-loops -finline-functions
CXXFLAGS  += -std=c++11 -fPIC -Wall -Wextra -fpermissive
CXXFLAGS  += -I$(OPENJP2_DIR) -I$(COMMON_DIR)
CXXFLAGS  += -DOPJ_STATIC
# Thread backend (WSL/Linux)
CXXFLAGS  += -DMUTEX_pthread
# OpenMP support
CXXFLAGS  += -fopenmp
LDFLAGS   += -pthread -fopenmp
LDFLAGS   += -lm

# J2K core + dependencies.
# Intentionally excluded:
# - JPIP indexing helpers
# - *_manager.c (JPIP indexing helpers)
# - bench_dwt.c, test_sparse_array.c (bench/tests)

SOURCES_ALG := \
  $(OPENJP2_DIR)/dwt.cpp \
  $(OPENJP2_DIR)/ht_dec.cpp \
  $(OPENJP2_DIR)/j2k.cpp \
  $(OPENJP2_DIR)/mct.cpp \
  $(OPENJP2_DIR)/mqc.cpp \
  $(OPENJP2_DIR)/t1.cpp \
  $(OPENJP2_DIR)/t1_generate_luts.cpp \
  $(OPENJP2_DIR)/t1_ht_generate_luts.cpp \
  $(OPENJP2_DIR)/t2.cpp \
  $(OPENJP2_DIR)/tcd.cpp

SOURCES_COMMON := \
  $(COMMON_DIR)/openjpeg.cpp \
  $(COMMON_DIR)/bio.cpp \
  $(COMMON_DIR)/cio.cpp \
  $(COMMON_DIR)/event.cpp \
  $(COMMON_DIR)/function_list.cpp \
  $(COMMON_DIR)/image.cpp \
  $(COMMON_DIR)/invert.cpp \
  $(COMMON_DIR)/opj_clock.cpp \
  $(COMMON_DIR)/opj_malloc.cpp \
  $(COMMON_DIR)/pi.cpp \
  $(COMMON_DIR)/sparse_array.cpp \
  $(COMMON_DIR)/tgt.cpp \
  $(COMMON_DIR)/thread.cpp

OBJECTS_ALG := $(patsubst $(OPENJP2_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES_ALG))
OBJECTS_COMMON := $(patsubst $(COMMON_DIR)/%.cpp,$(BUILD_DIR)/common/%.o,$(SOURCES_COMMON))

OBJECTS := $(OBJECTS_ALG) $(OBJECTS_COMMON)
TARGET  := $(BUILD_DIR)/libopenjp2_j2k.a
ENCODER := $(BUILD_DIR)/j2k_encode_pnm
DECODER := $(BUILD_DIR)/j2k_decode_pnm

.PHONY: all clean

all: $(TARGET) $(ENCODER) $(DECODER)

# Optional MPI version
mpi: $(BUILD_DIR)/j2k_encode_mpi
	@echo "MPI encoder built: $(BUILD_DIR)/j2k_encode_mpi"

$(TARGET): $(OBJECTS)
	$(AR) rcs $@ $^
	$(RANLIB) $@

$(ENCODER): $(BUILD_DIR)/j2k_encode_pnm.o $(TARGET)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lopenjp2_j2k $(LDFLAGS) -o $@

$(DECODER): $(BUILD_DIR)/j2k_decode_pnm.o $(TARGET)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lopenjp2_j2k $(LDFLAGS) -o $@

$(BUILD_DIR)/j2k_encode_mpi: j2k_encode_mpi.cpp $(TARGET)
	$(MPIXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lopenjp2_j2k $(LDFLAGS) -o $@

$(BUILD_DIR)/j2k_encode_profile: j2k_encode_profile.cpp $(TARGET)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lopenjp2_j2k $(LDFLAGS) -o $@

$(BUILD_DIR)/j2k_decode_profile: j2k_decode_profile.cpp $(TARGET)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lopenjp2_j2k $(LDFLAGS) -o $@

$(BUILD_DIR)/%.o: $(OPENJP2_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/common/%.o: $(COMMON_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/j2k_encode_pnm.o: j2k_encode_pnm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/j2k_decode_pnm.o: j2k_decode_pnm.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
