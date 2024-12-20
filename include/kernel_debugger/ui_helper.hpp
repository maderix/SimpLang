#ifndef KERNEL_DEBUGGER_UI_HELPER_HPP
#define KERNEL_DEBUGGER_UI_HELPER_HPP

#include <string>
#include <vector>
#include <functional>
#include <memory>

class UIHelper {
public:
    struct PromptOptions {
        std::string prompt;
        bool enableHistory;
        bool enableCompletion;
        size_t maxHistorySize;

        // Default constructor
        PromptOptions()
            : prompt("(simdy) ")
            , enableHistory(true)
            , enableCompletion(true)
            , maxHistorySize(1000)
        {}

        // Constructor with all parameters
        PromptOptions(const std::string& p, bool eHist, bool eComp, size_t maxHist)
            : prompt(p)
            , enableHistory(eHist)
            , enableCompletion(eComp)
            , maxHistorySize(maxHist)
        {}
    };

    explicit UIHelper(const PromptOptions& options = PromptOptions());

    // Input handling
    std::string getInput();
    void setCompletionCallback(std::function<std::vector<std::string>(const std::string&)> callback);
    
    // Display formatting
    void printSourceLine(const std::string& line, int lineNum, bool isCurrent);
    void printBreakpoint(int id, const std::string& location, const std::string& condition);
    void printError(const std::string& message);
    void printInfo(const std::string& message);
    void printWarning(const std::string& message);
    
    // History management
    void addToHistory(const std::string& input);
    std::string getPreviousHistory();
    std::string getNextHistory();
    
    // UI state
    void clearScreen();
    void setPrompt(const std::string& newPrompt);
    
private:
    PromptOptions options;
    std::vector<std::string> history;
    size_t historyIndex{0};
    std::function<std::vector<std::string>(const std::string&)> completionCallback;

    // Helper methods
    std::string handleTabCompletion(const std::string& partial);
    void initializeReadline();
    static void addColor(std::ostream& out, const std::string& color);
    static void resetColor(std::ostream& out);
};

#endif // KERNEL_DEBUGGER_UI_HELPER_HPP