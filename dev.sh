#!/bin/bash

# SimpleLang Development Helper Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "SimpleLang Development Helper"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup          - Initial setup (build container and compile)"
    echo "  build [--mlir] - Build the project (add --mlir for MLIR backend)"
    echo "  test           - Run standard tests"
    echo "  test-mlir      - Run MLIR tests"
    echo "  debug          - Start interactive debugger"
    echo "  shell          - Open development shell"
    echo "  clean          - Clean build artifacts"
    echo "  rebuild        - Clean and rebuild everything"
    echo ""
    echo "Examples:"
    echo "  $0 setup           # First time setup"
    echo "  $0 build           # Rebuild standard backend"
    echo "  $0 build --mlir    # Rebuild MLIR backend"
    echo "  $0 test-mlir       # Run MLIR test suite"
    echo "  $0 shell           # Interactive development"
}

ensure_container_built() {
    if ! docker images | grep -q simplang; then
        echo -e "${YELLOW}Building SimpleLang container...${NC}"
        docker-compose build simplang-dev
    fi
}

case "$1" in
    setup)
        echo -e "${BLUE}Setting up SimpleLang development environment...${NC}"
        docker-compose build simplang-dev
        echo -e "${YELLOW}Building project...${NC}"
        docker-compose run --rm simplang-dev ./build.sh --install-deps
        echo -e "${GREEN}Setup complete! You can now use:${NC}"
        echo "  $0 build         # Rebuild after changes"
        echo "  $0 build --mlir  # Build MLIR backend"
        echo "  $0 test          # Run tests"
        echo "  $0 test-mlir     # Run MLIR tests"
        echo "  $0 shell         # Interactive shell"
        ;;

    build)
        if [ "$2" = "--mlir" ]; then
            echo -e "${YELLOW}Building SimpleLang (MLIR)...${NC}"
            ensure_container_built
            docker-compose run --rm simplang-dev ./build.sh --mlir
        else
            echo -e "${YELLOW}Building SimpleLang (Standard)...${NC}"
            ensure_container_built
            docker-compose run --rm simplang-dev ./build.sh
        fi
        echo -e "${GREEN}Build complete!${NC}"
        ;;

    test)
        echo -e "${YELLOW}Running SimpleLang tests...${NC}"
        ensure_container_built
        docker-compose run --rm simplang-dev ./run_tests.sh
        ;;

    test-mlir)
        echo -e "${YELLOW}Running MLIR tests...${NC}"
        ensure_container_built
        docker-compose run --rm simplang-dev ./run_mlir_tests.sh
        ;;

    debug)
        echo -e "${BLUE}Starting SimpleLang debugger...${NC}"
        echo -e "${YELLOW}Type 'help' for commands, 'quit' to exit${NC}"
        ensure_container_built
        docker-compose run --rm simplang-debug
        ;;

    shell)
        echo -e "${BLUE}Starting development shell...${NC}"
        ensure_container_built
        docker-compose run --rm simplang-dev bash
        ;;

    clean)
        echo -e "${YELLOW}Cleaning build artifacts...${NC}"
        docker-compose run --rm simplang-dev rm -rf build build_mlir
        echo -e "${GREEN}Clean complete!${NC}"
        ;;

    rebuild)
        if [ "$2" = "--mlir" ]; then
            echo -e "${YELLOW}Rebuilding MLIR backend...${NC}"
            docker-compose run --rm simplang-dev rm -rf build_mlir
            docker-compose run --rm simplang-dev ./build.sh --mlir
        else
            echo -e "${YELLOW}Rebuilding standard backend...${NC}"
            docker-compose run --rm simplang-dev rm -rf build
            docker-compose run --rm simplang-dev ./build.sh
        fi
        echo -e "${GREEN}Rebuild complete!${NC}"
        ;;

    ""|help|-h|--help)
        print_usage
        ;;

    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac