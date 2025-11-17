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

# LLVM+MLIR paths - will be set after argument parsing
LLVM_PROJECT_DIR=""
LLVM_BUILD_DIR_ENV="${LLVM_BUILD_DIR}"  # Save env variable for later

print_usage() {
    echo "SimpleLang Build Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --check-deps           Check dependencies"
    echo "  --install-deps         Install missing dependencies"
    echo "  --setup-llvm           Build LLVM with MLIR and ARM target support (30-60 min)"
    echo "  --mlir                 Build MLIR backend (auto-builds LLVM if needed)"
    echo "  --all                  Build both standard and MLIR backends"
    echo "  --test                 Run tests after building"
    echo "  --clean                Clean build directories before building"
    echo "  --install-cross-tools  Install ARM cross-compilation tools (for linking)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  LLVM_BUILD_DIR         Path to existing LLVM build (overrides default)"
    echo ""
    echo "Examples:"
    echo "  $0 --install-deps                      # Install all dependencies"
    echo "  $0 --check-deps                        # Check dependencies"
    echo "  $0 --mlir --test                       # Build MLIR backend and run tests"
    echo "  $0 --setup-llvm                        # Build LLVM with ARM targets"
    echo "  LLVM_BUILD_DIR=/path/to/llvm ./build.sh --mlir  # Use custom LLVM"
    echo ""
    echo "Note: SimpleLang is built as an x86 binary that can generate code for"
    echo "      X86, ARM, and AArch64. Use --target flag when compiling .sl files."
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

install_dependencies() {
    echo -e "${BLUE}Installing dependencies...${NC}"

    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    elif [ "$(uname)" = "Darwin" ]; then
        OS="macos"
    else
        echo -e "${RED}Unknown OS. Please install dependencies manually.${NC}"
        return 1
    fi

    # Install based on OS
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        echo -e "${YELLOW}Installing dependencies for Ubuntu/Debian...${NC}"
        sudo apt-get update
        sudo apt-get install -y cmake ninja-build bison flex g++ git build-essential python3 libreadline-dev
    elif [ "$OS" = "macos" ]; then
        echo -e "${YELLOW}Installing dependencies for macOS...${NC}"
        if ! command -v brew >/dev/null 2>&1; then
            echo -e "${RED}Homebrew not found. Please install from https://brew.sh${NC}"
            return 1
        fi
        brew install cmake ninja bison flex git readline
    else
        echo -e "${RED}Unsupported OS: $OS${NC}"
        return 1
    fi

    echo -e "${GREEN}Dependencies installed!${NC}"
    return 0
}

install_cross_tools() {
    echo -e "${BLUE}Installing ARM cross-compilation tools...${NC}"

    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        OS_LIKE=$ID_LIKE
    else
        echo -e "${RED}Cross-compilation tools only available on Linux${NC}"
        return 1
    fi

    # Check if OS is Ubuntu-based or Debian-based
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ] || [[ "$OS_LIKE" == *"ubuntu"* ]] || [[ "$OS_LIKE" == *"debian"* ]]; then
        echo -e "${YELLOW}Installing ARM cross-compilers...${NC}"
        sudo apt-get update
        sudo apt-get install -y \
            gcc-aarch64-linux-gnu \
            g++-aarch64-linux-gnu \
            gcc-arm-linux-gnueabihf \
            g++-arm-linux-gnueabihf \
            qemu-user-static
        echo -e "${GREEN}ARM cross-compilation tools installed!${NC}"
        echo -e "${BLUE}Installed compilers:${NC}"
        echo "  - aarch64-linux-gnu-gcc (ARM 64-bit)"
        echo "  - arm-linux-gnueabihf-gcc (ARM 32-bit)"
        echo "  - qemu-user-static (for testing ARM binaries)"
    else
        echo -e "${RED}Unsupported OS: $OS${NC}"
        return 1
    fi

    return 0
}

setup_llvm_mlir() {
    local auto_mode=${1:-false}

    echo -e "${BLUE}Setting up LLVM with MLIR support...${NC}"

    # Check if LLVM is built
    if [ -f "$LLVM_BUILD_DIR/bin/mlir-opt" ]; then
        # If cross-compiling, verify ARM targets are available
        if [ -n "$TARGET_ARCH" ]; then
            if [ -f "$LLVM_BUILD_DIR/bin/llc" ]; then
                if ! "$LLVM_BUILD_DIR/bin/llc" --version | grep -q "aarch64\|arm"; then
                    echo -e "${YELLOW}LLVM found but missing ARM targets. Rebuilding...${NC}"
                else
                    echo -e "${GREEN}LLVM+MLIR already built with ARM support!${NC}"
                    echo -e "  Location: ${BLUE}$LLVM_BUILD_DIR${NC}"
                    return 0
                fi
            fi
        else
            echo -e "${GREEN}LLVM+MLIR already built!${NC}"
            echo -e "  Location: ${BLUE}$LLVM_BUILD_DIR${NC}"
            return 0
        fi
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
    # LLVM is built natively for x86 but with ARM target support enabled
    # This allows SimpleLang to generate code for both x86 and ARM architectures
    echo -e "${YELLOW}Building LLVM with MLIR and ARM target support...${NC}"
    echo -e "${BLUE}This enables SimpleLang (x86 binary) to generate ARM code${NC}"

    mkdir -p "$LLVM_BUILD_DIR"
    cd "$LLVM_BUILD_DIR"

    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="mlir;clang" \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_TARGETS_TO_BUILD="X86;ARM;AArch64" \
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

    mkdir -p "build_mlir"
    cd "build_mlir"

    cmake .. \
        -DLLVM_DIR="${LLVM_DIR}" \
        -DMLIR_DIR="${MLIR_DIR}" \
        -DUSE_MLIR=ON

    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) simplang
    cd "$SCRIPT_DIR"

    echo -e "${GREEN}MLIR build complete!${NC}"
    echo -e "  Compiler: ${BLUE}./build_mlir/src/simplang${NC}"
    echo -e "  ${GREEN}This x86 binary can generate code for X86, ARM, and AArch64${NC}"
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
INSTALL_DEPS=false
INSTALL_CROSS=false
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
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --install-cross-tools)
            INSTALL_CROSS=true
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

# Set LLVM paths after parsing arguments
if [ -z "$LLVM_BUILD_DIR_ENV" ]; then
    LLVM_PROJECT_DIR="$SCRIPT_DIR/llvm-project"
    LLVM_BUILD_DIR="$LLVM_PROJECT_DIR/build"
else
    LLVM_BUILD_DIR="$LLVM_BUILD_DIR_ENV"
    LLVM_PROJECT_DIR="$(dirname "$LLVM_BUILD_DIR")"
    echo -e "${GREEN}Using LLVM from: $LLVM_BUILD_DIR${NC}"
fi

# Execute based on flags
if $CHECK_DEPS; then
    check_dependencies
    exit $?
fi

if $INSTALL_DEPS; then
    install_dependencies
    exit $?
fi

if $INSTALL_CROSS; then
    install_cross_tools
    exit $?
fi

# Auto-check dependencies before building
if $BUILD_STANDARD || $BUILD_MLIR; then
    if ! check_dependencies 2>/dev/null; then
        echo ""
        read -p "Install missing dependencies? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_dependencies || {
                echo -e "${RED}Failed to install dependencies${NC}"
                exit 1
            }
        else
            echo -e "${RED}Cannot build without dependencies${NC}"
            exit 1
        fi
    fi
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
