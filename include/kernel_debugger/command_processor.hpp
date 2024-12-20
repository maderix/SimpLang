#ifndef KERNEL_DEBUGGER_COMMAND_PROCESSOR_HPP
#define KERNEL_DEBUGGER_COMMAND_PROCESSOR_HPP

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

class KernelDebugger;
class UIHelper;

class CommandProcessor {
public:
    struct Command {
        std::string name;
        std::string shortcut;
        std::string description;
        std::string usage;
        std::function<bool(const std::vector<std::string>&)> handler;
        
        Command(const std::string& n = "",
               const std::string& s = "",
               const std::string& d = "",
               const std::string& u = "",
               std::function<bool(const std::vector<std::string>&)> h = nullptr)
            : name(n), shortcut(s), description(d), usage(u), handler(h) {}
    };

    explicit CommandProcessor(KernelDebugger& debugger);
    ~CommandProcessor();

    // Main interface
    void run();
    bool processCommand(const std::string& cmdLine);
    
    // Command management
    void registerCommand(const Command& cmd);
    void removeCommand(const std::string& name);
    bool hasCommand(const std::string& name) const;
    
    // Help system
    void showHelp(const std::string& command = "") const;
    std::vector<std::string> getCompletions(const std::string& partial) const;

private:
    KernelDebugger& debugger;
    std::unique_ptr<UIHelper> ui;
    std::map<std::string, Command> commands;
    
    // Command registration
    void registerCommands();
    void setupCommandAliases();
    void setupUICallbacks();
    
    // Command handlers
    bool cmdHelp(const std::vector<std::string>& args);
    bool cmdFile(const std::vector<std::string>& args);
    bool cmdRun(const std::vector<std::string>& args);
    bool cmdContinue(const std::vector<std::string>& args);
    bool cmdBreak(const std::vector<std::string>& args);
    bool cmdDelete(const std::vector<std::string>& args);
    bool cmdEnable(const std::vector<std::string>& args);
    bool cmdDisable(const std::vector<std::string>& args);
    bool cmdStep(const std::vector<std::string>& args);
    bool cmdNext(const std::vector<std::string>& args);
    bool cmdFinish(const std::vector<std::string>& args);
    bool cmdList(const std::vector<std::string>& args);
    bool cmdInfo(const std::vector<std::string>& args);
    bool cmdPrint(const std::vector<std::string>& args);
    bool cmdRegisters(const std::vector<std::string>& args);
    bool cmdMemory(const std::vector<std::string>& args);
    bool cmdQuit(const std::vector<std::string>& args);
    
    // Helper methods
    std::vector<std::string> parseCommandLine(const std::string& cmdLine) const;
    bool validateArgCount(const std::vector<std::string>& args, size_t min, size_t max) const;
    std::string formatCommandHelp(const Command& cmd) const;
    void showAvailableCommands() const;
    void handleError(const std::string& message) const;
    
    // SIMD-specific helpers
    bool validateSimdRegister(const std::string& reg) const;
    void displaySimdRegister(const std::string& reg) const;
    void displayAllSimdRegisters() const;
};

#endif // KERNEL_DEBUGGER_COMMAND_PROCESSOR_HPP