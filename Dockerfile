# Multi-stage Dockerfile for SimpLang
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    llvm-14-dev \
    llvm-14-tools \
    clang-14 \
    cmake \
    libboost-all-dev \
    libreadline-dev \
    flex \
    bison \
    ninja-build \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up LLVM alternatives
RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-14 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Build the project
RUN ./build.sh

# Runtime stage
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    llvm-14-runtime \
    libboost-system1.74.0 \
    libreadline8 \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts
COPY --from=builder /app/build /app/build
COPY --from=builder /app/tests /app/tests
COPY --from=builder /app/run_tests.sh /app/

WORKDIR /app

# Default command
CMD ["./run_tests.sh"]