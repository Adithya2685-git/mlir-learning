get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
    ${dialect_libs}
    ${conversion_libs}
    ConvertMatMulToBlas
    GPT2SliceAnalysis
    GPT2DataFlowAnalysis
    DeadArgmaxElimination
    LinalgAnalysis
    ElementwiseFusion
    PolyhedralAnalysis
    PolyhedralParallelize
    MLIROptLib
    MLIRPass
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
)

add_llvm_executable(tutorial-opt tutorial-opt.cpp)

llvm_update_compile_flags(tutorial-opt)
target_link_libraries(tutorial-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tutorial-opt)