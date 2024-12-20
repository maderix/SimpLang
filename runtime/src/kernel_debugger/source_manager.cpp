#include "kernel_debugger/source_manager.hpp"
#include "kernel_debugger/debugger.hpp"
#include <fstream>
#include <iostream>
#include <regex>

bool SourceManager::loadSource(const std::string& filename) {
    std::cout << "Debug: Loading source file: " << filename << std::endl;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    
    sourceFiles[filename] = std::make_shared<SourceFile>(lines);
    
    std::cout << "Debug: Loaded " << lines.size() << " lines from " << filename << std::endl;
    return true;
}

bool SourceManager::setKernel(const std::string& filename) {
    if (loadSource(filename)) {
        currentFile = filename;
        return true;
    }
    return false;
}

void SourceManager::setLocation(const std::string& file, int line, const std::string& function) {
    if (!hasSource(file) || !validateLineNumber(file, line)) {
        std::cerr << "Invalid location: " << file << ":" << line << std::endl;
        return;
    }
    
    currentFile = file;
    currentFunction = function;
    sourceFiles[file]->currentLine = line;
}

SourceManager::Location SourceManager::getCurrentLocation() const {
    auto it = sourceFiles.find(currentFile);
    int line = (it != sourceFiles.end()) ? it->second->currentLine : 0;
    return Location(currentFile, line, currentFunction);
}

std::string SourceManager::getLine(const std::string& file, int line) const {
    auto it = sourceFiles.find(file);
    if (it == sourceFiles.end() || !validateLineNumber(file, line)) {
        return "";
    }
    return it->second->lines[line - 1];
}

int SourceManager::getLineCount(const std::string& file) const {
    auto it = sourceFiles.find(file);
    return it != sourceFiles.end() ? static_cast<int>(it->second->lines.size()) : 0;
}

void SourceManager::printLines(const std::string& file, int startLine, int count, std::ostream& out) const {
    std::cout << "DEBUG: SourceManager::printLines - start\n";
    std::cout << "DEBUG: file=" << file << ", startLine=" << startLine << ", count=" << count << "\n";
    
    auto it = sourceFiles.find(file);
    if (it == sourceFiles.end()) {
        std::cout << "DEBUG: Source file not found\n";
        return;
    }
    
    const auto& lines = it->second->lines;
    int start = std::max(1, startLine);
    int end = std::min(static_cast<int>(lines.size()), start + count - 1);
    
    std::cout << "DEBUG: Adjusted line range: " << start << " to " << end << "\n";
    std::cout << "DEBUG: Total lines in file: " << lines.size() << "\n";
    
    try {
        for (int i = start; i <= end; ++i) {
            out << std::setw(4) << i << " "
                << (i == it->second->currentLine ? ">" : " ")
                << " " << lines[i - 1];

            // Add value information for variable declarations and assignments
            const auto& line = lines[i - 1];
            if (line.find("var") != std::string::npos && line.find("=") != std::string::npos) {
                size_t varStart = line.find("var") + 3;
                size_t eqPos = line.find("=");
                std::string varName = line.substr(varStart, eqPos - varStart);
                varName = varName.substr(varName.find_first_not_of(" \t"));
                varName = varName.substr(0, varName.find_last_not_of(" \t") + 1);
                out << "    // " << varName << " = ...";
            }
            else if (line.find("return") != std::string::npos) {
                out << "    // Returning expression...";
            }
            
            out << "\n";
        }
    } catch (const std::exception& e) {
        std::cout << "DEBUG: Error printing lines: " << e.what() << "\n";
        throw;
    }
    std::cout << "DEBUG: SourceManager::printLines - end\n";
}

void SourceManager::showCurrentLocation(std::ostream& out) const {
    auto loc = getCurrentLocation();
    out << loc.file << ":" << loc.line;
    if (!loc.function.empty()) {
        out << " in " << loc.function;
    }
    out << "\n";
    
    // Show context (3 lines before and after current line)
    printLines(loc.file, std::max(1, loc.line - 3), 7, out);
}

bool SourceManager::isAtEnd() const {
    auto it = sourceFiles.find(currentFile);
    if (it == sourceFiles.end()) return true;
    return it->second->currentLine > static_cast<int>(it->second->lines.size());
}

void SourceManager::executeCurrentLine() {
    std::cout << "DEBUG: SourceManager::executeCurrentLine - start\n";
    
    if (!debugger || isAtEnd()) {
        std::cout << "DEBUG: No debugger or at end, returning\n";
        return;
    }

    auto currentLoc = getCurrentLocation();
    std::string line = getLine(currentFile, currentLoc.line);
    std::cout << "DEBUG: Executing line: " << line << "\n";

    // Parse and track variable declarations/assignments
    if (line.find("var") != std::string::npos) {
        parseDeclOrAssignment(line, true);  // true for declaration
    } else if (line.find("=") != std::string::npos) {
        parseDeclOrAssignment(line, false);  // false for assignment
    } else if (line.find("return") != std::string::npos) {
        std::cout << "DEBUG: Return statement: " << line << "\n";
        parseReturnStatement(line);
    }

    notifyDebugger("line", line);
    std::cout << "DEBUG: SourceManager::executeCurrentLine - end\n";
}

void SourceManager::parseDeclOrAssignment(const std::string& line, bool isDeclaration) {
    std::smatch match;
    std::regex pattern;
    
    if (isDeclaration) {
        pattern = std::regex(R"(var\s+(\w+)\s*=\s*([^;]+))");
    } else {
        pattern = std::regex(R"((\w+)\s*=\s*([^;]+))");
    }

    if (std::regex_search(line, match, pattern)) {
        std::string varName = match[1];
        std::string valueExpr = match[2];
        valueExpr = std::regex_replace(valueExpr, std::regex(R"(\s+)"), ""); // Remove whitespace
        
        // Notify debugger of variable update
        if (debugger) {
            debugger->getMemoryTracker().updateVariableValue(varName, valueExpr);
        }
    }
}

void SourceManager::parseReturnStatement(const std::string& line) {
    std::smatch match;
    std::regex pattern(R"(return\s+([^;]+))");
    if (std::regex_search(line, match, pattern)) {
        std::string expr = match[1];
        std::cout << "Returning: " << expr << "\n";
    }
}

void SourceManager::advanceLine() {
    auto it = sourceFiles.find(currentFile);
    if (it != sourceFiles.end() && !isAtEnd()) {
        it->second->currentLine++;
    }
}

void SourceManager::reset() {
    for (auto& [_, file] : sourceFiles) {
        file->currentLine = 1;
    }
    currentFunction.clear();
}

bool SourceManager::isFunctionStart(const std::string& line) const {
    static const std::regex functionPattern(R"(^\s*fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\()");
    return std::regex_search(line, functionPattern);
}

bool SourceManager::isSimdOperation(const std::string& line) const {
    return line.find("simd_") != std::string::npos;
}

std::string SourceManager::extractFunctionName(const std::string& line) const {
    static const std::regex functionNamePattern(R"(fn\s+([a-zA-Z_][a-zA-Z0-9_]*))");
    std::smatch match;
    if (std::regex_search(line, match, functionNamePattern)) {
        return match[1].str();
    }
    return "";
}

void SourceManager::parseAndExecuteLine(const std::string& line) {
    try {
        if (isFunctionStart(line)) {
            std::string funcName = extractFunctionName(line);
            currentFunction = funcName;
            notifyDebugger("enterFunction", funcName);
        }
        else if (isSimdOperation(line)) {
            notifyDebugger("simdOperation", line);
        }
        
        notifyDebugger("line", line);
        
    } catch (const std::exception& e) {
        std::cerr << "Error executing line: " << e.what() << std::endl;
    }
}

bool SourceManager::validateLineNumber(const std::string& file, int line) const {
    auto it = sourceFiles.find(file);
    return it != sourceFiles.end() && 
           line > 0 && 
           line <= static_cast<int>(it->second->lines.size());
}

void SourceManager::notifyDebugger(const std::string& event, const std::string& details) {
    if (debugger) {
        auto loc = getCurrentLocation();
        debugger->onDebugEvent(event, loc.file, loc.line, details);
    }
}