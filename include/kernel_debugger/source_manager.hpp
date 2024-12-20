#ifndef SOURCE_MANAGER_HPP
#define SOURCE_MANAGER_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <optional>
#include <sstream>
#include <iomanip>

class KernelDebugger;

class SourceManager {
public:
    struct SourceFile {
        std::vector<std::string> lines;
        int currentLine{1};
        bool hasBreakpoint{false};
        
        explicit SourceFile(const std::vector<std::string>& src) : lines(src) {}
    };

    struct Location {
        std::string file;
        int line;
        std::string function;
        
        Location(const std::string& f = "", int l = 0, const std::string& fn = "")
            : file(f), line(l), function(fn) {}
    };

    // Variable tracking
    struct Variable {
        std::string name;
        double value;
        
        Variable(const std::string& n = "", double v = 0.0)
            : name(n), value(v) {}
    };

    // Variable tracking methods
    void trackVariable(const std::string& name, double value) {
        variables[name] = Variable(name, value);
        std::cout << "DEBUG: Tracking variable " << name << " = " << value << "\n";
    }

    void updateVariableValue(const std::string& name, double value) {
        if (variables.find(name) != variables.end()) {
            variables[name].value = value;
            std::cout << "DEBUG: Updated variable " << name << " = " << value << "\n";
        }
    }

    std::optional<double> getVariableValue(const std::string& name) const {
        auto it = variables.find(name);
        if (it != variables.end()) {
            return it->second.value;
        }
        return std::nullopt;
    }
   void printVariables() const {
        std::cout << "Current variable values:\n";
        for (const auto& [name, var] : variables) {
            std::cout << "  " << name << " = " << var.value << "\n";
        }
    }

    explicit SourceManager(KernelDebugger* dbg = nullptr) : debugger(dbg) {}

    // File management
    bool loadSource(const std::string& filename);
    bool setKernel(const std::string& filename);
    std::string getCurrentKernel() const { return currentFile; }
    
    // Location management
    void setLocation(const std::string& file, int line, const std::string& function);
    Location getCurrentLocation() const;
    
    // Source access
    const std::map<std::string, std::shared_ptr<SourceFile>>& getSourceFiles() const { 
        return sourceFiles; 
    }
    
    bool hasSource(const std::string& file) const {
        return sourceFiles.find(file) != sourceFiles.end();
    }
    
    std::string getLine(const std::string& file, int line) const;
    int getLineCount(const std::string& file) const;
    
    // Display
    void printLines(const std::string& file, int startLine, int count, std::ostream& out) const;
    void showCurrentLocation(std::ostream& out = std::cout) const;
    
    // Execution control
    bool isAtEnd() const;
    void executeCurrentLine();
    void advanceLine();
    void reset();

    // Function analysis
    bool isFunctionStart(const std::string& line) const;
    bool isSimdOperation(const std::string& line) const;
    std::string extractFunctionName(const std::string& line) const;
    // Add public getter for current file
    std::string getCurrentFile() const { 
        return currentFile; 
    }

private:
    KernelDebugger* debugger{nullptr};
    std::string currentFile;
    std::map<std::string, std::shared_ptr<SourceFile>> sourceFiles;
    std::map<std::string, Variable> variables;
    std::string currentFunction;

    // Helper methods
    void parseAndExecuteLine(const std::string& line);
    bool validateLineNumber(const std::string& file, int line) const;
    void notifyDebugger(const std::string& event, const std::string& details = "");
    void parseDeclOrAssignment(const std::string& line, bool isDeclaration);
    void parseReturnStatement(const std::string& line);
};

#endif // SOURCE_MANAGER_HPP