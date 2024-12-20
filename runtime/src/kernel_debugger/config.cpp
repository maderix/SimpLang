// runtime/src/kernel_debugger/config.cpp
#include "kernel_debugger/config.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

bool DebuggerConfig::loadFromFile(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open config file: " << filename << std::endl;
            return false;
        }

        std::string line;
        std::string currentSection;

        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }

            // Check if this is a section header [section]
            if (line[0] == '[' && line.back() == ']') {
                currentSection = line.substr(1, line.size() - 2);
                continue;
            }

            // Split line into key = value
            auto pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                // Parse based on section
                if (currentSection == "display") {
                    if (key == "showLineNumbers") displayConfig.showLineNumbers = (value == "true");
                    else if (key == "colorOutput") displayConfig.colorOutput = (value == "true");
                    else if (key == "contextLines") displayConfig.contextLines = std::stoi(value);
                    else if (key == "showSourceOnBreak") displayConfig.showSourceOnBreak = (value == "true");
                    else if (key == "showRegistersOnBreak") displayConfig.showRegistersOnBreak = (value == "true");
                }
                else if (currentSection == "debug") {
                    if (key == "enableMemoryTracking") debugConfig.enableMemoryTracking = (value == "true");
                    else if (key == "enableSIMDTracking") debugConfig.enableSIMDTracking = (value == "true");
                    else if (key == "stopOnException") debugConfig.stopOnException = (value == "true");
                    else if (key == "logDebugEvents") debugConfig.logDebugEvents = (value == "true");
                    else if (key == "maxEventLogSize") debugConfig.maxEventLogSize = std::stoull(value);
                }
                else if (currentSection == "ui") {
                    if (key == "prompt") uiConfig.prompt = value;
                    else if (key == "enableHistory") uiConfig.enableHistory = (value == "true");
                    else if (key == "enableCompletion") uiConfig.enableCompletion = (value == "true");
                    else if (key == "maxHistorySize") uiConfig.maxHistorySize = std::stoull(value);
                }
            }
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}

bool DebuggerConfig::saveToFile(const std::string& filename) const {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open config file for writing: " << filename << std::endl;
            return false;
        }

        // Write display section
        file << "[display]\n";
        file << "showLineNumbers = " << (displayConfig.showLineNumbers ? "true" : "false") << "\n";
        file << "colorOutput = " << (displayConfig.colorOutput ? "true" : "false") << "\n";
        file << "contextLines = " << displayConfig.contextLines << "\n";
        file << "showSourceOnBreak = " << (displayConfig.showSourceOnBreak ? "true" : "false") << "\n";
        file << "showRegistersOnBreak = " << (displayConfig.showRegistersOnBreak ? "true" : "false") << "\n\n";

        // Write debug section
        file << "[debug]\n";
        file << "enableMemoryTracking = " << (debugConfig.enableMemoryTracking ? "true" : "false") << "\n";
        file << "enableSIMDTracking = " << (debugConfig.enableSIMDTracking ? "true" : "false") << "\n";
        file << "stopOnException = " << (debugConfig.stopOnException ? "true" : "false") << "\n";
        file << "logDebugEvents = " << (debugConfig.logDebugEvents ? "true" : "false") << "\n";
        file << "maxEventLogSize = " << debugConfig.maxEventLogSize << "\n\n";

        // Write ui section
        file << "[ui]\n";
        file << "prompt = " << uiConfig.prompt << "\n";
        file << "enableHistory = " << (uiConfig.enableHistory ? "true" : "false") << "\n";
        file << "enableCompletion = " << (uiConfig.enableCompletion ? "true" : "false") << "\n";
        file << "maxHistorySize = " << uiConfig.maxHistorySize << "\n";

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving config: " << e.what() << std::endl;
        return false;
    }
}

void DebuggerConfig::resetToDefaults() {
    displayConfig = DisplayConfig();
    debugConfig = DebugConfig();
    uiConfig = UIConfig();
}