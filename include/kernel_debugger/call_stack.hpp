#ifndef KERNEL_DEBUGGER_CALL_STACK_HPP
#define KERNEL_DEBUGGER_CALL_STACK_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <immintrin.h>

class CallStack {
public:
    struct Variable {
        enum class Type {
            DOUBLE,
            SSE_VECTOR,
            AVX_VECTOR,
            SSE_SLICE,
            AVX_SLICE
        };

        std::string name;
        Type type;
        void* value;
        
        Variable(const std::string& n, Type t, void* v) 
            : name(n), type(t), value(v) {}
    };

    struct Frame {
        std::string function;
        std::string file;
        int line;
        std::map<std::string, Variable> locals;
        
        Frame(const std::string& fn, const std::string& f, int l)
            : function(fn), file(f), line(l) {}
    };

    // Constructors
    CallStack() : inSimdOp(false) {}

    // Frame management
    void pushFrame(const std::string& function, const std::string& file, int line);
    void popFrame();
    void clear() { frames.clear(); }

    // Frame information
    size_t getDepth() const { return frames.size(); }
    bool isEmpty() const { return frames.empty(); }
    const Frame* getCurrentFrame() const;

    // Location management
    void updateLocation(const std::string& file, int line);

    // SIMD operations
    void enterSimdOp(const std::string& op);
    void exitSimdOp();

    // Variable management
    void addLocal(const std::string& name, Variable::Type type, void* value);
    void updateLocal(const std::string& name, void* value);
    const Variable* getLocal(const std::string& name) const;

    // Output functions
    void printBacktrace(std::ostream& out = std::cout) const;
    void printLocals(std::ostream& out = std::cout) const;

private:
    std::vector<Frame> frames;
    bool inSimdOp;
    std::string currentSimdOp;

    std::string formatVariable(const Variable& var) const;
};

#endif // KERNEL_DEBUGGER_CALL_STACK_HPP