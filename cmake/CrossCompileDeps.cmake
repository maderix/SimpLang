# Dummy targets for missing cross-compile dependencies
# LLVM/MLIR expect these to exist, but they're not needed for cross-compilation

if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB INTERFACE IMPORTED)
    message(STATUS "Created dummy ZLIB::ZLIB target for cross-compilation")
endif()

if(NOT TARGET Terminfo::terminfo)
    add_library(Terminfo::terminfo INTERFACE IMPORTED)
    message(STATUS "Created dummy Terminfo::terminfo target for cross-compilation")
endif()
