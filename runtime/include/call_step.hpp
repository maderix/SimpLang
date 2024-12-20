#ifndef CALL_STACK_HPP
#define CALL_STACK_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>

class CallStackManager {
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

    struct StackFrame {
        std::string function;
        std::string file;
        int line;
        std::map<std::string, Variable> locals;
        
        StackFrame(const std::string& fn, const std::string& f, int l)
            : function(fn), file(f), line(l) {}
    };

private:
    std::vector<StackFrame> frames;
    bool inSimdOp;
    std::string currentSimdOp;

public:
    CallStackManager() : inSimdOp(false) {}

    // Frame management
    void pushFrame(const std::string& function, const std::string& file, int line);
    void popFrame();
    void updateLocation(const std::string& file, int line);
    
    // SIMD operation tracking
    void enterSimdOp(const std::string& op);
    void exitSimdOp();
    
    // Variable tracking
    void addLocal(const std::string& name, Variable::Type type, void* value);
    void updateLocal(const std::string& name, void* value);
    const Variable* getLocal(const std::string& name) const;
    
    // Stack inspection
    const StackFrame* getCurrentFrame() const;
    std::vector<StackFrame> getFrames() const { return frames; }
    bool isInSimdOp() const { return inSimdOp; }
    std::string getCurrentSimdOp() const { return currentSimdOp; }

    // Stack trace output
    void printBacktrace(std::ostream& out) const;
    void printLocals(std::ostream& out) const;
    std::string formatVariable(const Variable& var) const;
};

#endif // CALL_STACK_HPP