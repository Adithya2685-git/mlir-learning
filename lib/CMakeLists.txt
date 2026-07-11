add_mlir_library(ConvertMatMulToBlas
    ConvertMatMulToBlas.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/ConvertMatMulToBlas

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRPass
)

add_mlir_library(GPT2SliceAnalysis
    GPT2SliceAnalysis.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRAnalysis
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRTensorDialect
)

add_mlir_library(GPT2DataFlowAnalysis
    GPT2DataFlowAnalysis.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRAnalysis
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRTensorDialect
)

add_mlir_library(DeadArgmaxElimination
    DeadArgmaxElimination.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRAnalysis
    MLIRFuncDialect
    MLIRLinalgDialect
)

add_mlir_library(LinalgAnalysis
    LinalgAnalysis.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRTensorDialect
)

add_mlir_library(ElementwiseFusion
    ElementwiseFusion.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTransformUtils
)

add_mlir_library(PolyhedralAnalysis
    PolyhedralAnalysis.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRAnalysis
    MLIRAffineDialect
    MLIRAffineAnalysis
    MLIRLinalgDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRPresburger
    MLIRSupport
)

add_mlir_library(PolyhedralParallelize
    PolyhedralParallelize.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRAnalysis
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRLinalgTransformOps
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRPresburger
    MLIRTransformUtils
    MLIRRewrite
)
