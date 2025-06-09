#!/bin/bash
set -e

echo "Setting up SimpLang development environment..."

# Update package lists
sudo apt-get update

# Install LLVM 14 and development tools
sudo apt-get install -y \
    llvm-14-dev \
    llvm-14-tools \
    clang-14 \
    clang-tools-14 \
    libc++-14-dev \
    libc++abi-14-dev \
    libboost-all-dev \
    libreadline-dev \
    flex \
    bison \
    gdb \
    valgrind \
    ninja-build \
    pkg-config

# Set up symlinks for LLVM tools
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-14 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100

# Verify installations
echo "Verifying installations..."
llvm-config --version
cmake --version
clang --version

echo "Development environment setup complete!"
echo "You can now run: ./build.sh to build the project"