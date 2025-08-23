#include "logger.hpp"

// Initialize static members
LogLevel Logger::currentLevel = LogLevel::INFO;  // Default to INFO level
bool Logger::colorEnabled = true;  // Enable colors by default (can be disabled for pipes/files)