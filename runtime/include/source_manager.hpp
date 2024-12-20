#ifndef SOURCE_MANAGER_HPP
#define SOURCE_MANAGER_HPP

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>

// Forward declarations
class KernelDebugger;

class SourceManager {
public:
    struct SourceFile {
        std::string path;
        std::vector<std::string> lines;
        std::set<int> breakpoints;
        int currentLine;
        
        SourceFile() : currentLine(0) {}
    };

    struct SourceLocation {
        std::string file;
        int line;
        std::string function;
        
        SourceLocation() : line(0) {}
        SourceLocation(const std::string& f, int l, const std::string& fn)
            : file(f), line(l), function(fn) {}
    };

private:
    friend class KernelDebugger;
    std::map<std::string, std::shared_ptr<SourceFile>> sourceFiles;
    SourceLocation currentLocation;
    std::string currentKernel;

public:
    // File management
    bool loadSource(const std::string& path);
    void unloadSource(const std::string& path);
    bool hasSource(const std::string& path) const;
    
    // Source navigation
    const SourceFile* getSource(const std::string& path) const;
    const std::map<std::string, std::shared_ptr<SourceFile>>& getSourceFiles() const { 
        return sourceFiles; 
    }
    std::string getCurrentKernel() const { return currentKernel; }
    SourceLocation getCurrentLocation() const { return currentLocation; }
    
    // Line access and formatting
    std::string getLine(const std::string& file, int lineNo) const;
    std::vector<std::string> getLines(const std::string& file, int start, int count) const;
    void printLines(const std::string& file, int start, int count, std::ostream& out) const;
    
    // Location tracking
    void setLocation(const std::string& file, int line, const std::string& function);
    void setKernel(const std::string& kernel) { currentKernel = kernel; }
    
    // Breakpoint management
    bool addBreakpoint(const std::string& file, int line);
    bool removeBreakpoint(const std::string& file, int line);
    bool hasBreakpoint(const std::string& file, int line) const;
    std::vector<std::pair<std::string, int>> getAllBreakpoints() const;

private:
    // Helper functions
    static std::string readFile(const std::string& path);
    static std::vector<std::string> splitLines(const std::string& content);
    static std::string trimLine(const std::string& line);
    bool isValidSourceFile(const std::string& content) const;
};

#endif // SOURCE_MANAGER_HPP