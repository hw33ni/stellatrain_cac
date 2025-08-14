FROM nvcr.io/nvidia/pytorch:25.02-py3

RUN apt-get update -qq && apt-get install -y libsodium-dev libbsd-dev python3-dev git bc

# check python version
RUN echo "Python version (FORCED REBUILD):" && python3 --version && date

# Setup Users
RUN useradd --create-home --user-group stellatrain

# Copy Files
WORKDIR /home/stellatrain
COPY --chown=stellatrain:stellatrain . explore-dp

# Configure
WORKDIR /home/stellatrain/explore-dp/backend/libzmq
RUN ./autogen.sh

WORKDIR /home/stellatrain/explore-dp/backend
# RUN cmake -DPYTHON_EXECUTABLE=/usr/bin/python \
#       -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch \
#       -DPYTHON_LIB_PATH=/usr/local/lib/python3.10 \
#       -DPYTHON_INCLUDE_PATH=/usr/include/python3.10 \
#       -DPYTHON_VERSION=3.10 \
#       -B build

# WORKDIR /home/stellatrain/explore-dp/backend # Ensure this WORKDIR is active

# RUN pip install --no-cache-dir "pybind11>=2.12" \
#       && echo "pybind11 $(python -m pip show pybind11 | grep -i version)" \
#       && git -C /home/stellatrain/explore-dp/backend/pybind11 fetch --tags \
#       && git -C /home/stellatrain/explore-dp/backend/pybind11 checkout v2.13.6

# after the COPY and before the first cmake invocation
RUN apt-get update -qq && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /home/stellatrain/explore-dp/backend/pybind11 \
 && mkdir  /home/stellatrain/explore-dp/backend/pybind11 \
 && curl -L https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz \
      | tar xz --strip-components=1 -C /home/stellatrain/explore-dp/backend/pybind11

RUN cmake \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch \
      -DPYTHON_LIB_PATH=/usr/local/lib/python3.12 \
      -DPYTHON_INCLUDE_PATH=/usr/include/python3.12 \
      -DPYTHON_VERSION=3.12 \
      -B build

# RUN \
#     # Determine Python specific paths
#     PYTHON_EXECUTABLE=$(which python3) && \
#     PYTHON_VERSION_SHORT=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
#     PYTHON_VERSION_FULL=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')") && \
#     PIP_TORCH_LOCATION=$(pip3 show torch | grep Location | awk '{print $2}') && \
#     TORCH_CMAKE_DIR="${PIP_TORCH_LOCATION}/torch/share/cmake/Torch" && \
#     PYTHON_SYS_LIBDIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") && \
#     PYTHON_SYS_LDLIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))") && \
#     PYTHON_SYS_INCLUDEPY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))") && \
#     echo "--- CMake Build Configuration ---" && \
#     echo "Python Executable: ${PYTHON_EXECUTABLE}" && \
#     echo "Python Version (Short): ${PYTHON_VERSION_SHORT}" && \
#     echo "Python Version (Full): ${PYTHON_VERSION_FULL}" && \
#     echo "PyTorch Location (pip): ${PIP_TORCH_LOCATION}" && \
#     echo "Torch CMake Dir: ${TORCH_CMAKE_DIR}" && \
#     echo "Python System Lib Dir: ${PYTHON_SYS_LIBDIR}" && \
#     echo "Python System LDLibrary: ${PYTHON_SYS_LDLIBRARY}" && \
#     echo "Python System IncludePY Dir: ${PYTHON_SYS_INCLUDEPY}" && \
#     echo "---------------------------------" && \
#     cmake \
#       -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
#       -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_DIR}" \
#       # For modern CMake finding Python, these are preferred if your CMakeLists.txt uses find_package(Python ...)
#       -DPython_EXECUTABLE=${PYTHON_EXECUTABLE} \
#       -DPython_LIBRARIES="${PYTHON_SYS_LIBDIR}/${PYTHON_SYS_LDLIBRARY}" \
#       -DPython_INCLUDE_DIRS="${PYTHON_SYS_INCLUDEPY}" \
#       # Your custom variables if CMakeLists.txt specifically uses them
#       -DPYTHON_LIB_PATH="${PIP_TORCH_LOCATION}/torch/lib" \ 
#       # Often torch ships its own relevant .so files here or links to system
#       -DPYTHON_INCLUDE_PATH="${PYTHON_SYS_INCLUDEPY}" \     
#       # Standard Python includes
#       -DPYTHON_VERSION="${PYTHON_VERSION_SHORT}" \        
#       # e.g., "3.12"
#       -B build

# Build
RUN cmake --build build --config RelWithDebInfo --target all -j

# Install
WORKDIR /home/stellatrain/explore-dp/backend/scikit-optimize
RUN pip install --editable .

USER stellatrain
WORKDIR /home/stellatrain/explore-dp