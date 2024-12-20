#include "kernel_debugger/ui_helper.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <readline/readline.h>
#include <readline/history.h>

// ANSI color codes
namespace Color {
    const char* RESET   = "\033[0m";
    const char* RED     = "\033[31m";
    const char* GREEN   = "\033[32m";
    const char* YELLOW  = "\033[33m";
    const char* BLUE    = "\033[34m";
    const char* MAGENTA = "\033[35m";
    const char* CYAN    = "\033[36m";
}

// Static completion callback for readline
static std::function<std::vector<std::string>(const std::string&)>* g_completionCallback = nullptr;

static char* completion_generator(const char* text, int state) {
    static std::vector<std::string> matches;
    static size_t match_index = 0;
    
    if (state == 0) {
        matches.clear();
        match_index = 0;
        
        if (g_completionCallback) {
            matches = (*g_completionCallback)(text);
        }
    }
    
    if (match_index >= matches.size()) {
        return nullptr;
    }
    
    return strdup(matches[match_index++].c_str());
}

static char** command_completion(const char* text, int start, int end) {
    rl_attempted_completion_over = 1;
    return rl_completion_matches(text, completion_generator);
}

UIHelper::UIHelper(const PromptOptions& options) : options(options) {
    // Initialize readline
    rl_attempted_completion_function = command_completion;
    
    if (options.enableHistory) {
        using_history();
    }
}

std::string UIHelper::getInput() {
    char* line = readline(options.prompt.c_str());
    if (!line) {
        return "";
    }
    
    std::string input(line);
    free(line);
    
    // Add to history if non-empty and history is enabled
    if (!input.empty() && options.enableHistory) {
        add_history(input.c_str());
        if (history_length > static_cast<int>(options.maxHistorySize)) {
            HIST_ENTRY* removed = remove_history(0);
            if (removed) {
                free_history_entry(removed);
            }
        }
    }
    
    return input;
}

void UIHelper::setCompletionCallback(std::function<std::vector<std::string>(const std::string&)> callback) {
    g_completionCallback = &callback;
}

void UIHelper::printSourceLine(const std::string& line, int lineNum, bool isCurrent) {
    std::cout << std::setw(4) << lineNum << " "
              << (isCurrent ? ">" : " ") << " ";
              
    if (isCurrent) {
        std::cout << Color::CYAN;
    }
    
    std::cout << line << Color::RESET << "\n";
}

void UIHelper::printBreakpoint(int id, const std::string& location, const std::string& condition) {
    std::cout << Color::YELLOW << "Breakpoint " << id << Color::RESET
              << " at " << location;
    if (!condition.empty()) {
        std::cout << " when " << condition;
    }
    std::cout << "\n";
}

void UIHelper::printError(const std::string& message) {
    std::cerr << Color::RED << "Error: " << message << Color::RESET << "\n";
}

void UIHelper::printInfo(const std::string& message) {
    std::cout << Color::GREEN << message << Color::RESET << "\n";
}

void UIHelper::printWarning(const std::string& message) {
    std::cout << Color::YELLOW << "Warning: " << message << Color::RESET << "\n";
}

void UIHelper::addToHistory(const std::string& input) {
    if (options.enableHistory && !input.empty()) {
        add_history(input.c_str());
    }
}

std::string UIHelper::getPreviousHistory() {
    HIST_ENTRY* entry = current_history();
    if (entry) {
        return entry->line;
    }
    return "";
}

std::string UIHelper::getNextHistory() {
    HIST_ENTRY* entry = next_history();
    if (entry) {
        return entry->line;
    }
    return "";
}

void UIHelper::clearScreen() {
    std::cout << "\033[2J\033[H";  // ANSI escape sequence to clear screen and move cursor to top
}

void UIHelper::setPrompt(const std::string& newPrompt) {
    options.prompt = newPrompt;
}

void UIHelper::addColor(std::ostream& out, const std::string& color) {
    out << color;
}

void UIHelper::resetColor(std::ostream& out) {
    out << Color::RESET;
}