#!/bin/bash
set -e

echo "Setting up SimpLang development environment..."

# Update package lists
sudo apt-get update

# Install LLVM 21 and development tools
sudo apt-get install -y \
    llvm-21-dev \
    llvm-21-tools \
    clang-21 \
    clang-tools-21 \
    libc++-21-dev \
    libc++abi-21-dev \
    libboost-all-dev \
    libreadline-dev \
    flex \
    bison \
    gdb \
    valgrind \
    ninja-build \
    pkg-config

# Set up symlinks for LLVM tools
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-21 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100

# Verify installations
echo "Verifying installations..."
llvm-config --version
cmake --version
clang --version

echo "Development environment setup complete!"
echo "You can now run: ./build.sh to build the project"