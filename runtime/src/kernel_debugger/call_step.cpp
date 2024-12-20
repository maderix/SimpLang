#include "call_stack.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <immintrin.h>

void CallStackManager::pushFrame(const std::string& function, const std::string& file, int line) {
    frames.emplace_back(function, file, line);
}

void CallStackManager::popFrame() {
    if (!frames.empty()) {
        frames.pop_back();
    }
}

void CallStackManager::updateLocation(const std::string& file, int line) {
    if (!frames.empty()) {
        frames.back().file = file;
        frames.back().line = line;
    }
}

void CallStackManager::enterSimdOp(const std::string& op) {
    inSimdOp = true;
    currentSimdOp = op;
    pushFrame(op, getCurrentFrame()->file, getCurrentFrame()->line);
}

void CallStackManager::exitSimdOp() {
    if (inSimdOp) {
        inSimdOp = false;
        currentSimdOp.clear();
        popFrame();
    }
}

void CallStackManager::addLocal(const std::string& name, Variable::Type type, void* value) {
    if (!frames.empty()) {
        frames.back().locals.emplace(name, Variable(name, type, value));
    }
}

void CallStackManager::updateLocal(const std::string& name, void* value) {
    if (!frames.empty()) {
        auto& locals = frames.back().locals;
        auto it = locals.find(name);
        if (it != locals.end()) {
            it->second.value = value;
        }
    }
}

const CallStackManager::Variable* CallStackManager::getLocal(const std::string& name) const {
    if (frames.empty()) return nullptr;
    
    auto& locals = frames.back().locals;
    auto it = locals.find(name);
    return it != locals.end() ? &it->second : nullptr;
}

const CallStackManager::StackFrame* CallStackManager::getCurrentFrame() const {
    return frames.empty() ? nullptr : &frames.back();
}

void CallStackManager::printBacktrace(std::ostream& out) const {
    out << "\nBacktrace:\n";
    
    int frameNum = 0;
    for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
        out << "#" << frameNum++ << " " 
            << it->function << " at "
            << it->file << ":" << it->line << "\n";
    }
}

void CallStackManager::printLocals(std::ostream& out) const {
    if (frames.empty()) {
        out << "No active frame\n";
        return;
    }
    
    out << "\nLocal variables in " << frames.back().function << ":\n";
    
    for (const auto& [name, var] : frames.back().locals) {
        out << "  " << std::left << std::setw(15) << name 
            << " = " << formatVariable(var) << "\n";
    }
}

std::string CallStackManager::formatVariable(const Variable& var) const {
    std::stringstream ss;
    
    switch (var.type) {
        case Variable::Type::DOUBLE: {
            double value = *static_cast<double*>(var.value);
            ss << value;
            break;
        }
        case Variable::Type::SSE_VECTOR: {
            __m256d* vec = static_cast<__m256d*>(var.value);
            alignas(32) double values[4];
            _mm256_store_pd(values, *vec);
            ss << "[" << values[0];
            for (int i = 1; i < 4; i++) {
                ss << ", " << values[i];
            }
            ss << "]";
            break;
        }
        case Variable::Type::AVX_VECTOR: {
            __m512d* vec = static_cast<__m512d*>(var.value);
            alignas(64) double values[8];
            _mm512_store_pd(values, *vec);
            ss << "[" << values[0];
            for (int i = 1; i < 8; i++) {
                ss << ", " << values[i];
            }
            ss << "]";
            break;
        }
        case Variable::Type::SSE_SLICE:
        case Variable::Type::AVX_SLICE:
            ss << "<slice>";
            break;
    }
    
    return ss.str();
}