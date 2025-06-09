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
    echo "  setup     - Initial setup (build container and compile)"
    echo "  build     - Build the project"
    echo "  test      - Run all tests"
    echo "  debug     - Start interactive debugger"
    echo "  shell     - Open development shell"
    echo "  clean     - Clean build artifacts"
    echo "  rebuild   - Clean and rebuild everything"
    echo ""
    echo "Examples:"
    echo "  $0 setup           # First time setup"
    echo "  $0 build           # Rebuild after code changes"
    echo "  $0 test            # Run test suite"
    echo "  $0 debug           # Start debugger"
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
        docker-compose run --rm simplang-dev ./build.sh
        echo -e "${GREEN}Setup complete! You can now use:${NC}"
        echo "  $0 build    # Rebuild after changes"
        echo "  $0 test     # Run tests"
        echo "  $0 debug    # Start debugger"
        echo "  $0 shell    # Interactive shell"
        ;;
    
    build)
        echo -e "${YELLOW}Building SimpleLang...${NC}"
        ensure_container_built
        docker-compose run --rm simplang-dev ./build.sh
        echo -e "${GREEN}Build complete!${NC}"
        ;;
    
    test)
        echo -e "${YELLOW}Running SimpleLang tests...${NC}"
        ensure_container_built
        docker-compose run --rm simplang-test
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
        docker-compose run --rm simplang-dev
        ;;
    
    clean)
        echo -e "${YELLOW}Cleaning build artifacts...${NC}"
        docker-compose run --rm simplang-dev rm -rf build/*
        echo -e "${GREEN}Clean complete!${NC}"
        ;;
    
    rebuild)
        echo -e "${YELLOW}Rebuilding everything...${NC}"
        docker-compose run --rm simplang-dev rm -rf build/*
        docker-compose run --rm simplang-dev ./build.sh
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