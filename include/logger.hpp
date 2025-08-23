#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>
#include <sstream>

enum class LogLevel {
    ERROR = 0,
    WARNING = 1,
    INFO = 2,
    DEBUG = 3,
    TRACE = 4
};

class Logger {
private:
    static LogLevel currentLevel;
    static bool colorEnabled;
    
    static const char* getLevelString(LogLevel level) {
        switch (level) {
            case LogLevel::ERROR:   return "ERROR";
            case LogLevel::WARNING: return "WARN ";
            case LogLevel::INFO:    return "INFO ";
            case LogLevel::DEBUG:   return "DEBUG";
            case LogLevel::TRACE:   return "TRACE";
            default:                return "?????";
        }
    }
    
    static const char* getColorCode(LogLevel level) {
        if (!colorEnabled) return "";
        switch (level) {
            case LogLevel::ERROR:   return "\033[31m";  // Red
            case LogLevel::WARNING: return "\033[33m";  // Yellow
            case LogLevel::INFO:    return "\033[32m";  // Green
            case LogLevel::DEBUG:   return "\033[36m";  // Cyan
            case LogLevel::TRACE:   return "\033[90m";  // Gray
            default:                return "";
        }
    }
    
    static const char* getResetCode() {
        return colorEnabled ? "\033[0m" : "";
    }
    
public:
    static void setLevel(LogLevel level) {
        currentLevel = level;
    }
    
    static LogLevel getLevel() {
        return currentLevel;
    }
    
    static void setColorEnabled(bool enabled) {
        colorEnabled = enabled;
    }
    
    static void setLevelFromString(const std::string& levelStr) {
        std::string upper = levelStr;
        for (auto& c : upper) c = toupper(c);
        
        if (upper == "ERROR" || upper == "0") {
            currentLevel = LogLevel::ERROR;
        } else if (upper == "WARNING" || upper == "WARN" || upper == "1") {
            currentLevel = LogLevel::WARNING;
        } else if (upper == "INFO" || upper == "2") {
            currentLevel = LogLevel::INFO;
        } else if (upper == "DEBUG" || upper == "3") {
            currentLevel = LogLevel::DEBUG;
        } else if (upper == "TRACE" || upper == "4") {
            currentLevel = LogLevel::TRACE;
        } else {
            currentLevel = LogLevel::INFO;  // Default
        }
    }
    
    template<typename... Args>
    static void log(LogLevel level, Args... args) {
        if (level > currentLevel) return;
        
        std::ostream& out = (level == LogLevel::ERROR) ? std::cerr : std::cout;
        out << getColorCode(level) << "[" << getLevelString(level) << "] " << getResetCode();
        
        std::ostringstream oss;
        ((oss << args), ...);
        out << oss.str() << std::endl;
    }
    
    template<typename... Args>
    static void error(Args... args) {
        log(LogLevel::ERROR, args...);
    }
    
    template<typename... Args>
    static void warning(Args... args) {
        log(LogLevel::WARNING, args...);
    }
    
    template<typename... Args>
    static void info(Args... args) {
        log(LogLevel::INFO, args...);
    }
    
    template<typename... Args>
    static void debug(Args... args) {
        log(LogLevel::DEBUG, args...);
    }
    
    template<typename... Args>
    static void trace(Args... args) {
        log(LogLevel::TRACE, args...);
    }
};

// Convenience macros that include file and line information for debugging
#define LOG_ERROR(...) Logger::error(__VA_ARGS__)
#define LOG_WARNING(...) Logger::warning(__VA_ARGS__)
#define LOG_INFO(...) Logger::info(__VA_ARGS__)
#define LOG_DEBUG(...) Logger::debug(__VA_ARGS__)
#define LOG_TRACE(...) Logger::trace(__VA_ARGS__)

// LLVM-specific logging that preserves the existing llvm::errs() output
#define LOG_LLVM(level, ...) do { \
    if (Logger::getLevel() >= level) { \
        llvm::errs() << __VA_ARGS__; \
    } \
} while(0)

#endif // LOGGER_HPP