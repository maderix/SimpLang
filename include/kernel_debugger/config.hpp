#ifndef KERNEL_DEBUGGER_CONFIG_HPP
#define KERNEL_DEBUGGER_CONFIG_HPP

#include <string>
#include <map>
#include <memory>

class DebuggerConfig {
public:
    // Configuration groups
    struct DisplayConfig {
        bool showLineNumbers{true};
        bool colorOutput{true};
        int contextLines{3};
        bool showSourceOnBreak{true};
        bool showRegistersOnBreak{false};
    };

    struct DebugConfig {
        bool enableMemoryTracking{true};
        bool enableSIMDTracking{true};
        bool stopOnException{true};
        bool logDebugEvents{true};
        size_t maxEventLogSize{1000};
    };

    struct UIConfig {
        std::string prompt{"(simdy) "};
        bool enableHistory{true};
        bool enableCompletion{true};
        size_t maxHistorySize{1000};
    };

    // Singleton access
    static std::shared_ptr<DebuggerConfig> getInstance() {
        static std::shared_ptr<DebuggerConfig> instance = std::shared_ptr<DebuggerConfig>(new DebuggerConfig);
        return instance;
    }

    // Config access
    DisplayConfig& display() { return displayConfig; }
    DebugConfig& debug() { return debugConfig; }
    UIConfig& ui() { return uiConfig; }
    
    const DisplayConfig& display() const { return displayConfig; }
    const DebugConfig& debug() const { return debugConfig; }
    const UIConfig& ui() const { return uiConfig; }

    // Load/Save configuration
    bool loadFromFile(const std::string& filename);
    bool saveToFile(const std::string& filename) const;
    
    // Reset configuration
    void resetToDefaults();
private:
    DebuggerConfig() = default;
    
    DisplayConfig displayConfig;
    DebugConfig debugConfig;
    UIConfig uiConfig;

    // Make config non-copyable
    DebuggerConfig(const DebuggerConfig&) = delete;
    DebuggerConfig& operator=(const DebuggerConfig&) = delete;
};

#endif // KERNEL_DEBUGGER_CONFIG_HPP