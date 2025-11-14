#!/bin/bash
set -e

# Get script directory (works from any location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# LLVM+MLIR paths
LLVM_PROJECT_DIR="$SCRIPT_DIR/llvm-project"
LLVM_BUILD_DIR="$LLVM_PROJECT_DIR/build"

print_usage() {
    echo "SimpleLang Build Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --check-deps     Check and install dependencies"
    echo "  --setup-llvm     Build LLVM with MLIR support (takes 30-60 min)"
    echo "  --mlir           Build MLIR backend"
    echo "  --all            Build both standard and MLIR backends"
    echo "  --test           Run tests after building"
    echo "  --clean          Clean build directories before building"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --check-deps         # Check dependencies"
    echo "  $0 --setup-llvm         # Build LLVM+MLIR (first time only)"
    echo "  $0 --mlir --test        # Build MLIR backend and run tests"
    echo "  $0 --all --test         # Build everything and run tests"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

init_submodules() {
    if [ -f ".gitmodules" ] && [ -d ".git" ]; then
        echo -e "${YELLOW}Initializing git submodules...${NC}"
        git submodule update --init --recursive
        echo -e "${GREEN}Submodules initialized!${NC}"
    fi
}

check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"

    local missing_deps=()

    # Check required tools
    check_command cmake || missing_deps+=("cmake")
    check_command ninja || missing_deps+=("ninja-build")
    check_command bison || missing_deps+=("bison")
    check_command flex || missing_deps+=("flex")
    check_command g++ || missing_deps+=("g++")
    check_command git || missing_deps+=("git")

    # Check for readline library
    if ldconfig -p 2>/dev/null | grep -q libreadline; then
        echo -e "${GREEN}✓${NC} libreadline found"
    else
        echo -e "${RED}✗${NC} libreadline not found"
        missing_deps+=("libreadline-dev")
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Missing dependencies:${NC} ${missing_deps[*]}"
        echo ""
        echo "Install on Ubuntu/Debian:"
        echo "  sudo apt-get install ${missing_deps[*]} build-essential python3"
        echo ""
        echo "Install on macOS:"
        echo "  brew install ${missing_deps[*]}"
        echo ""
        return 1
    fi

    echo -e "${GREEN}All dependencies satisfied!${NC}"
    return 0
}

setup_llvm_mlir() {
    local auto_mode=${1:-false}

    echo -e "${BLUE}Setting up LLVM with MLIR support...${NC}"

    if [ -f "$LLVM_BUILD_DIR/bin/mlir-opt" ]; then
        echo -e "${GREEN}LLVM+MLIR already built!${NC}"
        echo -e "  Location: ${BLUE}$LLVM_BUILD_DIR${NC}"
        return 0
    fi

    if [ "$auto_mode" = "false" ]; then
        echo -e "${YELLOW}This will take 30-60 minutes and requires ~20GB disk space.${NC}"
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            return 1
        fi
    else
        echo -e "${YELLOW}Building LLVM+MLIR (30-60 min, ~20GB)...${NC}"
    fi

    # Clone LLVM project if not exists
    if [ ! -d "$LLVM_PROJECT_DIR" ]; then
        echo -e "${YELLOW}Cloning LLVM project...${NC}"
        git clone --depth 1 --branch llvmorg-14.0.0 https://github.com/llvm/llvm-project.git "$LLVM_PROJECT_DIR"
    fi

    # Build LLVM with MLIR
    echo -e "${YELLOW}Building LLVM with MLIR (this will take a while)...${NC}"
    mkdir -p "$LLVM_BUILD_DIR"
    cd "$LLVM_BUILD_DIR"

    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="mlir;clang" \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_TARGETS_TO_BUILD="X86" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INSTALL_UTILS=ON

    ninja -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    cd "$SCRIPT_DIR"
    echo -e "${GREEN}LLVM+MLIR build complete!${NC}"
    echo -e "  Location: ${BLUE}$LLVM_BUILD_DIR${NC}"
}

clean_build() {
    echo -e "${YELLOW}Cleaning build directories...${NC}"
    rm -rf build build_mlir
    echo -e "${GREEN}Clean complete!${NC}"
}

build_standard() {
    echo -e "${BLUE}Building SimpleLang (Standard LLVM Backend)...${NC}"

    # Initialize submodules if needed
    init_submodules

    # Use system LLVM
    if command -v llvm-config-14 >/dev/null 2>&1; then
        LLVM_DIR="$(llvm-config-14 --cmakedir)"
    elif command -v llvm-config >/dev/null 2>&1; then
        LLVM_DIR="$(llvm-config --cmakedir)"
    else
        echo -e "${RED}Error: llvm-config not found${NC}"
        echo "Install LLVM: sudo apt-get install llvm-14-dev"
        return 1
    fi

    mkdir -p build
    cd build
    cmake .. -DLLVM_DIR="${LLVM_DIR}" -DUSE_MLIR=OFF
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd "$SCRIPT_DIR"

    echo -e "${GREEN}Standard build complete!${NC}"
    echo -e "  Compiler: ${BLUE}./build/src/simplang${NC}"
}

build_mlir() {
    echo -e "${BLUE}Building SimpleLang (MLIR Backend)...${NC}"

    # Initialize submodules if needed
    init_submodules

    # Check if LLVM+MLIR is available, if not, build it
    if [ ! -f "$LLVM_BUILD_DIR/bin/mlir-opt" ]; then
        echo -e "${YELLOW}LLVM+MLIR not found. Building it now...${NC}"
        setup_llvm_mlir true || {
            echo -e "${RED}Failed to setup LLVM+MLIR${NC}"
            return 1
        }
    fi

    LLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm"
    MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"

    mkdir -p build_mlir
    cd build_mlir
    cmake .. \
        -DLLVM_DIR="${LLVM_DIR}" \
        -DMLIR_DIR="${MLIR_DIR}" \
        -DUSE_MLIR=ON
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) simplang
    cd "$SCRIPT_DIR"

    echo -e "${GREEN}MLIR build complete!${NC}"
    echo -e "  Compiler: ${BLUE}./build_mlir/src/simplang${NC}"
}

run_standard_tests() {
    echo -e "${BLUE}Running standard tests...${NC}"
    if [ -f "./run_tests.sh" ]; then
        ./run_tests.sh
    else
        echo -e "${YELLOW}Warning: run_tests.sh not found${NC}"
    fi
}

run_mlir_tests() {
    echo -e "${BLUE}Running MLIR tests...${NC}"
    if [ -f "./run_mlir_tests.sh" ]; then
        ./run_mlir_tests.sh
    else
        echo -e "${YELLOW}Warning: run_mlir_tests.sh not found${NC}"
    fi
}

# Parse arguments
CHECK_DEPS=false
SETUP_LLVM=false
BUILD_STANDARD=false
BUILD_MLIR=false
RUN_TESTS=false
CLEAN=false

if [ $# -eq 0 ]; then
    # Default: build standard only
    BUILD_STANDARD=true
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --check-deps)
            CHECK_DEPS=true
            shift
            ;;
        --setup-llvm)
            SETUP_LLVM=true
            shift
            ;;
        --mlir)
            BUILD_MLIR=true
            shift
            ;;
        --all)
            BUILD_STANDARD=true
            BUILD_MLIR=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Execute based on flags
if $CHECK_DEPS; then
    check_dependencies
    exit $?
fi

if $SETUP_LLVM; then
    setup_llvm_mlir
    exit $?
fi

if $CLEAN; then
    clean_build
fi

if $BUILD_STANDARD; then
    build_standard
    if $RUN_TESTS; then
        run_standard_tests
    fi
fi

if $BUILD_MLIR; then
    build_mlir
    if $RUN_TESTS; then
        run_mlir_tests
    fi
fi

echo ""
echo -e "${GREEN}Build process complete!${NC}"
