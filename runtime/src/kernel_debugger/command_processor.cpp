#include "kernel_debugger/command_processor.hpp"
#include "kernel_debugger/debugger.hpp"
#include "kernel_debugger/ui_helper.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>

CommandProcessor::CommandProcessor(KernelDebugger& dbg) 
    : debugger(dbg), ui(std::make_unique<UIHelper>()) {
    registerCommands();
    //setupCommandAliases();
    //setupUICallbacks();
}

CommandProcessor::~CommandProcessor() = default;

void CommandProcessor::run() {
    ui->printInfo("SIMDY Debugger v1.0");
    ui->printInfo("Type 'help' for list of commands.");
    
    bool running = true;
    while (running) {
        std::string input = ui->getInput();
        if (input.empty()) continue;
        
        running = processCommand(input);
    }
}

void CommandProcessor::registerCommands() {
    commands["help"] = Command("help", "h", "Display help for commands", 
                             "help [command]",
                             [this](const auto& args) { return cmdHelp(args); });
    
    commands["file"] = Command("file", "f", "Load a kernel file",
                             "file <filename>",
                             [this](const auto& args) { return cmdFile(args); });
    
    commands["run"] = Command("run", "r", "Start or restart program execution",
                            "run",
                            [this](const auto& args) { return cmdRun(args); });
    
    commands["continue"] = Command("continue", "c", "Continue program execution",
                                 "continue",
                                 [this](const auto& args) { return cmdContinue(args); });
    
    commands["break"] = Command("break", "b", "Set breakpoint",
                              "break <line> [if <condition>]",
                              [this](const auto& args) { return cmdBreak(args); });
    
    commands["delete"] = Command("delete", "d", "Delete breakpoint",
                               "delete <breakpoint-id>",
                               [this](const auto& args) { return cmdDelete(args); });
    
    commands["enable"] = Command("enable", "en", "Enable breakpoint",
                               "enable <breakpoint-id>",
                               [this](const auto& args) { return cmdEnable(args); });
    
    commands["disable"] = Command("disable", "dis", "Disable breakpoint",
                                "disable <breakpoint-id>",
                                [this](const auto& args) { return cmdDisable(args); });
    
    commands["step"] = Command("step", "s", "Step program until it reaches a different source line",
                             "step",
                             [this](const auto& args) { return cmdStep(args); });
    
    commands["next"] = Command("next", "n", "Step program, proceeding through subroutine calls",
                             "next",
                             [this](const auto& args) { return cmdNext(args); });
    
    commands["finish"] = Command("finish", "fin", "Execute until selected stack frame returns",
                               "finish",
                               [this](const auto& args) { return cmdFinish(args); });
    
    commands["list"] = Command("list", "l", "List source code",
                             "list [line]",
                             [this](const auto& args) { return cmdList(args); });
    
    commands["info"] = Command("info", "i", "Display program state",
                             "info <what>",
                             [this](const auto& args) { return cmdInfo(args); });
    
    commands["print"] = Command("print", "p", "Print value of expression",
                              "print <expression>",
                              [this](const auto& args) { return cmdPrint(args); });
    
    commands["registers"] = Command("registers", "reg", "Display SIMD registers",
                                  "registers [register-name]",
                                  [this](const auto& args) { return cmdRegisters(args); });
    
    commands["memory"] = Command("memory", "mem", "Display memory contents",
                               "memory <address> [count]",
                               [this](const auto& args) { return cmdMemory(args); });
    
    commands["quit"] = Command("quit", "q", "Exit the debugger",
                             "quit",
                             [this](const auto& args) { return cmdQuit(args); });
}

void CommandProcessor::setupCommandAliases() {
    // Add shortcuts to commands map
    for (const auto& [name, cmd] : commands) {
        if (!cmd.shortcut.empty()) {
            commands[cmd.shortcut] = cmd;
        }
    }
}

void CommandProcessor::setupUICallbacks() {
    ui->setCompletionCallback([this](const std::string& partial) {
        return getCompletions(partial);
    });
}

void CommandProcessor::registerCommand(const Command& cmd) {
    commands[cmd.name] = cmd;
    if (!cmd.shortcut.empty()) {
        commands[cmd.shortcut] = cmd;
    }
}

void CommandProcessor::removeCommand(const std::string& name) {
    auto it = commands.find(name);
    if (it != commands.end()) {
        if (!it->second.shortcut.empty()) {
            commands.erase(it->second.shortcut);
        }
        commands.erase(it);
    }
}

bool CommandProcessor::hasCommand(const std::string& name) const {
    return commands.find(name) != commands.end();
}

bool CommandProcessor::processCommand(const std::string& cmdLine) {
    auto args = parseCommandLine(cmdLine);
    if (args.empty()) return true;

    auto cmdIt = commands.find(args[0]);
    if (cmdIt == commands.end()) {
        ui->printError("Unknown command: " + args[0]);
        return true;
    }

    try {
        return cmdIt->second.handler(args);
    } catch (const std::exception& e) {
        ui->printError("Command failed: " + std::string(e.what()));
        return true;
    }
}

std::vector<std::string> CommandProcessor::parseCommandLine(const std::string& cmdLine) const {
    std::vector<std::string> args;
    bool inQuotes = false;
    std::string current;
    
    for (char c : cmdLine) {
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ' ' && !inQuotes) {
            if (!current.empty()) {
                args.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    
    if (!current.empty()) {
        args.push_back(current);
    }
    
    return args;
}

bool CommandProcessor::validateArgCount(const std::vector<std::string>& args, 
                                     size_t min, size_t max) const {
    if (args.size() < min) {
        ui->printError("Too few arguments for " + args[0]);
        showHelp(args[0]);
        return false;
    }
    if (max != size_t(-1) && args.size() > max) {
        ui->printError("Too many arguments for " + args[0]);
        showHelp(args[0]);
        return false;
    }
    return true;
}

void CommandProcessor::showHelp(const std::string& command) const {
    if (command.empty()) {
        ui->printInfo("Available commands:");
        for (const auto& [name, cmd] : commands) {
            if (name == cmd.name) {  // Only show primary commands, not shortcuts
                ui->printInfo(formatCommandHelp(cmd));
            }
        }
        return;
    }

    auto it = commands.find(command);
    if (it == commands.end()) {
        ui->printError("Unknown command: " + command);
        return;
    }

    ui->printInfo(formatCommandHelp(it->second));
}

std::string CommandProcessor::formatCommandHelp(const Command& cmd) const {
    std::stringstream ss;
    ss << std::left << std::setw(15) << cmd.name;
    if (!cmd.shortcut.empty()) {
        ss << "(" << cmd.shortcut << ")";
        ss << std::setw(10-cmd.shortcut.length()) << "";
    } else {
        ss << std::setw(12) << "";
    }
    ss << cmd.description << "\n";
    ss << "Usage: " << cmd.usage;
    return ss.str();
}

// Command Handlers
bool CommandProcessor::cmdHelp(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 1, 2)) return true;
    showHelp(args.size() > 1 ? args[1] : "");
    return true;
}

bool CommandProcessor::cmdFile(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 2, 2)) return true;
    
    if (debugger.loadKernel(args[1])) {
        ui->printInfo("Loaded kernel file: " + args[1]);
    } else {
        ui->printError("Failed to load kernel file: " + args[1]);
    }
    return true;
}

bool CommandProcessor::cmdRun(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 1, 1)) return true;
    debugger.start();
    debugger.continueExecution();
    return true;
}

bool CommandProcessor::cmdContinue(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 1, 1)) return true;
    debugger.continueExecution();
    return true;
}

bool CommandProcessor::cmdBreak(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 2, 4)) return true;
    
    try {
        std::string file = debugger.getCurrentFile();
        int line = std::stoi(args[1]);
        std::string condition;
        
        if (args.size() > 2) {
            if (args[2] != "if") {
                ui->printError("Expected 'if' for breakpoint condition");
                return true;
            }
            condition = args[3];
        }
        
        int id = debugger.addBreakpoint(file, line, condition);
        if (id >= 0) {
            ui->printInfo("Breakpoint " + std::to_string(id) + " set at line " + 
                         args[1] + (condition.empty() ? "" : " if " + condition));
        } else {
            ui->printError("Failed to set breakpoint");
        }
    } catch (const std::exception& e) {
        ui->printError("Invalid line number");
    }
    return true;
}


bool CommandProcessor::cmdDelete(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 2, 2)) return true;
    
    try {
        int id = std::stoi(args[1]);
        if (debugger.removeBreakpoint(id)) {
            ui->printInfo("Breakpoint " + std::to_string(id) + " deleted");
        } else {
            ui->printError("No breakpoint number " + args[1]);
        }
    } catch (const std::exception& e) {
        ui->printError("Invalid breakpoint number");
    }
    return true;
}

bool CommandProcessor::cmdEnable(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 2, 2)) return true;
    
    try {
        int id = std::stoi(args[1]);
        debugger.enableBreakpoint(id, true);  // Removed if condition since it returns void
        ui->printInfo("Breakpoint " + std::to_string(id) + " enabled");
    } catch (const std::exception& e) {
        ui->printError("Invalid breakpoint number");
    }
    return true;
}

bool CommandProcessor::cmdDisable(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 2, 2)) return true;
    
    try {
        int id = std::stoi(args[1]);
        debugger.enableBreakpoint(id, false);  // Removed if condition since it returns void
        ui->printInfo("Breakpoint " + std::to_string(id) + " disabled");
    } catch (const std::exception& e) {
        ui->printError("Invalid breakpoint number");
    }
    return true;
}

bool CommandProcessor::cmdStep(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 1, 1)) return true;
    debugger.stepIn();
    return true;
}

bool CommandProcessor::cmdNext(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 1, 1)) return true;
    debugger.stepOver();
    return true;
}

bool CommandProcessor::cmdFinish(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 1, 1)) return true;
    debugger.stepOut();
    return true;
}

bool CommandProcessor::cmdList(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 1, 2)) return true;
    
    int line = -1;
    if (args.size() == 2) {
        try {
            line = std::stoi(args[1]);
        } catch (const std::exception& e) {
            ui->printError("Invalid line number");
            return true;
        }
    }
    
    std::string currentFile = debugger.getCurrentFile();
    debugger.showSource(currentFile, line, 10);
    return true;
}



// In displayAllSimdRegisters function:
void CommandProcessor::displayAllSimdRegisters() const {
    debugger.printVectorState();
}

bool CommandProcessor::cmdInfo(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 2, 2)) return true;
    
    const std::string& what = args[1];
    if (what == "breakpoints") {
        debugger.listBreakpoints();
    } else if (what == "registers") {
        displayAllSimdRegisters();
    } else if (what == "locals") {
        debugger.printLocals();
    } else if (what == "stack") {
        debugger.printBacktrace();
    } else {
        ui->printError("Unknown info command. Available: breakpoints, registers, locals, stack");
    }
    return true;
}

bool CommandProcessor::cmdPrint(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 2, size_t(-1))) return true;
    
    std::string expr;
    for (size_t i = 1; i < args.size(); ++i) {
        if (i > 1) expr += " ";
        expr += args[i];
    }
    
    debugger.printExpression(expr);
    return true;
}

bool CommandProcessor::cmdRegisters(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 1, 2)) return true;
    
    if (args.size() == 1) {
        displayAllSimdRegisters();
    } else {
        displaySimdRegister(args[1]);
    }
    return true;
}

bool CommandProcessor::cmdMemory(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 2, 3)) return true;
    
    try {
        void* addr = reinterpret_cast<void*>(std::stoull(args[1], nullptr, 16));
        size_t count = args.size() > 2 ? std::stoull(args[2]) : 16;
        debugger.displayMemory(addr, count);
    } catch (const std::exception& e) {
        ui->printError("Invalid memory address or count");
    }
    return true;
}

bool CommandProcessor::cmdQuit(const std::vector<std::string>& args) {
    if (!validateArgCount(args, 1, 1)) return true;
    ui->printInfo("Quitting debugger");
    return false;  // Signal to exit
}

bool CommandProcessor::validateSimdRegister(const std::string& reg) const {
    static const std::vector<std::string> validRegs = {
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
        "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7"
    };
    return std::find(validRegs.begin(), validRegs.end(), reg) != validRegs.end();
}

void CommandProcessor::displaySimdRegister(const std::string& reg) const {
    if (!validateSimdRegister(reg)) {
        ui->printError("Invalid SIMD register: " + reg);
        return;
    }
    debugger.printVectorState();
}

std::vector<std::string> CommandProcessor::getCompletions(const std::string& partial) const {
    std::vector<std::string> matches;
    for (const auto& [name, cmd] : commands) {
        if (name.find(partial) == 0 && name == cmd.name) {  // Only complete primary commands
            matches.push_back(name);
        }
    }
    return matches;
}

void CommandProcessor::handleError(const std::string& message) const {
    ui->printError(message);
}