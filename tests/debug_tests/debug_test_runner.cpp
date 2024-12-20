#include "kernel_debugger/debugger.hpp"
#include "kernel_debugger/ui_helper.hpp"
#include "kernel.h"
#include <iostream>
#include <sstream>
#include <readline/readline.h>
#include <readline/history.h>
#include <signal.h>

// Forward declaration
class SimpleLangDebugger;

// Global variables for signal handling
static volatile bool keep_running = true;
static SimpleLangDebugger* active_debugger = nullptr;

// Function declarations
void cleanup_terminal() {
    // Don't call readline functions directly in signal handler
    std::cout << std::endl;
}

void sigint_handler(int) {
    std::cout << "\nDEBUG: Received SIGINT\n";
    keep_running = false;
    cleanup_terminal();
}

// Command completion function
static char* command_generator(const char* text, int state) {
    static const char* commands[] = {
        "help", "file", "list", "break", "run", "step", 
        "continue", "print", "quit", "h", "f", "l", "b", 
        "r", "s", "c", "p", "q", nullptr
    };
    static int list_index, len;
    
    if (!state) {
        list_index = 0;
        len = strlen(text);
    }
    
    while (commands[list_index]) {
        if (strncmp(commands[list_index], text, len) == 0) {
            return strdup(commands[list_index++]);
        }
        list_index++;
    }
    
    return nullptr;
}

// Completion function for readline
static char** debugger_completion(const char* text, int start, int end) {
    (void)end;  // Unused
    
    rl_attempted_completion_over = 1;
    
    if (start == 0) {
        return rl_completion_matches(text, command_generator);
    }
    
    return nullptr;
}

class SimpleLangDebugger {
private:
    KernelDebugger& debugger;
    bool running;
    std::map<std::string, std::string> breakpoints;

    void initializeReadline() {
        std::cout << "DEBUG: Initializing readline\n";
        
        // Initialize history
        using_history();
        
        // Set up completion
        rl_readline_name = "simdy";  // Add this line
        rl_attempted_completion_function = debugger_completion;
        
        // Don't use special characters for completion
        rl_basic_word_break_characters = const_cast<char*>(" \t\n\"\\'`@$><=;|&{(");
        
        // Don't modify the terminal directly
        rl_catch_signals = 0;
        rl_catch_sigwinch = 0;
        rl_deprep_term_function = nullptr;
        rl_prep_term_function = nullptr;
        
        std::cout << "DEBUG: Readline initialized\n";
    }

    void cleanupReadline() {
        std::cout << "DEBUG: Cleaning up readline\n";
        clear_history();
        std::cout << "DEBUG: Readline cleanup complete\n";
    }

    std::vector<std::string> tokenize(const std::string& line) {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        while (ss >> token) {
            tokens.push_back(token);
        }
        return tokens;
    }

    void handleFileCommand(const std::vector<std::string>& tokens) {
        std::cout << "DEBUG: Handling file command\n";
        if (tokens.size() < 2) {
            std::cout << "Usage: file <kernel.sl>\n";
            return;
        }
        std::cout << "DEBUG: Loading file " << tokens[1] << "\n";
        if (debugger.loadKernel(tokens[1])) {
            std::cout << "Loaded " << tokens[1] << "\n";
            debugger.showSource("", -5, 10);
        } else {
            std::cout << "Failed to load " << tokens[1] << "\n";
        }
        std::cout << "DEBUG: File command complete\n";
    }

    void handleListCommand(const std::vector<std::string>& tokens) {
        std::cout << "DEBUG: Handling list command\n";
        try {
            int line = tokens.size() > 1 ? std::stoi(tokens[1]) : -5;
            debugger.showSource("", line, 10);
        } catch (...) {
            debugger.showSource("", -5, 10);
        }
        std::cout << "DEBUG: List command complete\n";
    }

    void handleBreakCommand(const std::vector<std::string>& tokens) {
        std::cout << "DEBUG: Handling break command\n";
        if (tokens.size() < 2) {
            std::cout << "Usage: break <line>\n";
            return;
        }

        try {
            int line = std::stoi(tokens[1]);
            int id = debugger.addBreakpoint(debugger.getCurrentFile(), line);
            std::cout << "Breakpoint " << id << " set at line " << line << "\n";
        } catch (...) {
            std::cout << "Invalid line number\n";
        }
        std::cout << "DEBUG: Break command complete\n";
    }

    void handleRunCommand() {
        std::cout << "DEBUG: Handling run command - start\n";
        std::cout << "DEBUG: Current file: " << debugger.getCurrentFile() << "\n";
        std::cout << "DEBUG: About to call debugger.start()\n";
        debugger.start();
        std::cout << "DEBUG: Called debugger.start()\n";

        // Execute program
        std::cout << "DEBUG: About to continue execution\n";
        debugger.continueExecution();
        std::cout << "DEBUG: Execution completed\n";

        std::cout << "DEBUG: About to show source\n";
        debugger.showSource("", -5, 10);
        std::cout << "DEBUG: Source shown\n";
        std::cout << "DEBUG: Run command complete\n";
    }

    void handleStepCommand() {
        std::cout << "DEBUG: Handling step command\n";
        debugger.stepIn();
        std::cout << "Stepping...\n";
        debugger.showSource("", -5, 10);
        std::cout << "DEBUG: Step command complete\n";
    }

    void handleContinueCommand() {
        std::cout << "DEBUG: Handling continue command\n";
        std::cout << "Continuing execution...\n";
        debugger.continueExecution();
        debugger.showSource("", -5, 10);
        std::cout << "DEBUG: Continue command complete\n";
    }

    void handlePrintCommand(const std::vector<std::string>& tokens) {
        std::cout << "DEBUG: Handling print command\n";
        if (tokens.size() < 2) {
            std::cout << "Usage: print <variable>\n";
            return;
        }
        debugger.printExpression(tokens[1]);
        std::cout << "DEBUG: Print command complete\n";
    }

public:
    SimpleLangDebugger() 
        : debugger(KernelDebugger::getInstance())
        , running(false) 
    {
        std::cout << "DEBUG: SimpleLangDebugger constructor begin\n";
        initializeReadline();
        debugger.initialize();
        std::cout << "DEBUG: Debugger initialized\n";
        debugger.start();
        std::cout << "DEBUG: Debugger started\n";
        std::cout << "DEBUG: SimpleLangDebugger constructor complete\n";
    }

    ~SimpleLangDebugger() {
        cleanupReadline();
    }

    void run() {
        std::cout << "DEBUG: Starting debugger run loop\n";
        std::cout << "SimDy Debugger v1.0\n";
        std::cout << "Type 'help' for a list of commands.\n\n";
        std::cout.flush();

        running = true;
        while (running && keep_running) {
            std::cout << "DEBUG: Waiting for command\n";
            std::cout.flush();
            
            char* line = readline("(simdy) ");
            if (!line) {
                std::cout << "\nDEBUG: Got EOF or error\n";
                break;
            }

            std::string input{line};
            free(line);

            std::cout << "DEBUG: Got input: '" << input << "'\n";

            if (input.empty()) {
                continue;
            }

            add_history(input.c_str());

            std::cout << "DEBUG: Processing command: " << input << "\n";
            try {
                std::vector<std::string> tokens = tokenize(input);
                if (tokens.empty()) {
                    std::cout << "DEBUG: No tokens, continuing\n";
                    continue;
                }

                std::string cmd = tokens[0];
                std::cout << "DEBUG: Command is: '" << cmd << "'\n";

                if (cmd == "quit" || cmd == "q") {
                    std::cout << "DEBUG: Got quit command\n";
                    running = false;
                }
                else if (cmd == "help" || cmd == "h") {
                    std::cout << "Available commands:\n"
                             << "  file <filename>   - Load source file\n"
                             << "  list [n]          - List source code\n"
                             << "  break <n>         - Set breakpoint at line n\n"
                             << "  run               - Run program\n"
                             << "  step              - Step one line\n"
                             << "  continue          - Continue execution\n"
                             << "  print <var>       - Print variable value\n"
                             << "  quit              - Exit debugger\n";
                }
                else if (cmd == "file" || cmd == "f") {
                    handleFileCommand(tokens);
                }
                else if (cmd == "list" || cmd == "l") {
                    handleListCommand(tokens);
                }
                else if (cmd == "break" || cmd == "b") {
                    handleBreakCommand(tokens);
                }
                else if (cmd == "run" || cmd == "r") {
                    handleRunCommand();
                }
                else if (cmd == "step" || cmd == "s") {
                    handleStepCommand();
                }
                else if (cmd == "continue" || cmd == "c") {
                    handleContinueCommand();
                }
                else if (cmd == "print" || cmd == "p") {
                    handlePrintCommand(tokens);
                }
                else {
                    std::cout << "Unknown command: '" << cmd << "'\n";
                }
            }
            catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
            std::cout << "DEBUG: Command processing complete\n";
        }

        std::cout << "DEBUG: Exiting debugger run loop\n";
    }
};



int main() {
    try {
        std::cout << "DEBUG: Starting main\n";
        
        signal(SIGINT, sigint_handler);
        std::cout << "DEBUG: Set up signal handler\n";

        SimpleLangDebugger debugger;
        active_debugger = &debugger;
        std::cout << "DEBUG: Created debugger instance\n";
        
        debugger.run();
        std::cout << "DEBUG: Debugger run completed\n";

        active_debugger = nullptr;
        return 0;
    }
    catch (const std::exception& e) {
        if (active_debugger) {
            cleanup_terminal();
        }
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}