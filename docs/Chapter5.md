# Chapter 5: Integration of OpenBLAS for Matrix Multiplications

We will now link to OpenBLAS that executes our matrix multiplications. For that, we define a custom MLIR pass that converts high-level matrix multiplication operations into optimized library calls. We first need to install OpenBLAS, for example by downloading the [latest release](https://github.com/OpenMathLib/OpenBLAS/releases) version: currently OpenBLAS-0.3.31.tar.gz. Then unpack with `tar -xvzf OpenBLAS-0.3.31.tar.gz`. You can find the docs for installing OpenBLAS on Linux [here](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/install.md#linux). The [user_manual](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/user_manual.md#compiling-openblas) describes how to compile OpenBLAS then. We run `make -j$(nproc)` and then `make install`. As described in the article, the CPU architecture is automatically detected by default when running *make*. The libraries are in /opt/OpenBLAS by default. However, you can also specify the directory: `make PREFIX=/path/to/installation/directory install`, e.g., `make PREFIX=ml-compiler-exercise/openblas install`. Later when compiling, we can use `-L../../openblas/lib -lopenblas` to link the OpenBLAS library. Now, we need to tell the dynamic linker about it. For example, you can do that by adding a custom library path system-wide:

```
echo "/ml-compiler-exercise/openblas/lib/" | sudo tee /etc/ld.so.conf.d/openblas.conf
sudo ldconfig
```

or temporary: `export LD_LIBRARY_PATH=/ml-compiler-exercise/openblas/lib/:$LD_LIBRARY_PATH`

We will now write our own pass that converts matrix multiplications (linalg matmuls). The pass is `-convert-matmul-to-blas`. In [tests/](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/tests/matmul_to_blas.mlir) you can run an example.
The header file [ConvertMatMulToBlas.h](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/lib/ConvertMatMulToBlas.h) and the implementation [ConvertMatMulToBlas.cpp](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/lib/ConvertMatMulToBlas.cpp) is in `lib/`. 

In the header file we define our class:  

The core of this file is the ConvertMatmulToBlasLibraryCallPass class. As you proably already now, a pass in MLIR is essentially a transformation that visits your program and modifies it. This pass inherits from PassWrapper, which is MLIR's template for creating passes.
The template parameter `OperationPass<mlir::ModuleOp>` tells MLIR that this pass operates on entire modules (your complete program). The pass will visit every operation in the module and decide what to transform.

Key Methods
- getDependentDialects: This tells MLIR which dialects your pass will use. When the pass runs, it needs to know about LLVM, function, memref, and arithmetic operations since it will be generating these as part of the conversion.
- runOnOperation: This is where the actual transformation happens (the implementation lives in a separate .cpp file). This method gets called by the MLIR framework when the pass executes.
- getArgument: Returns the command-line flag name (-convert-matmul-to-blas) used to invoke this pass.
- getDescription: A human-readable explanation of what the pass does.

So basically, whenever this pass encounters a linalg.matmul operation (a generic matrix multiply), it replaces it with a direct call to OpenBLAS. This is a common optimization because OpenBLAS implementations are highly tuned and often much faster than generic linear algebra code.


Now let's see how the pass actually works! This implementation file is where all the transformation logic lives.

The Conversion Pattern:

The core of this pass is the MatmulOpToBlasLibraryCall class, which inherits from ConversionPattern. Think of a pattern as a reusable rule: "whenever you see a linalg.matmul operation, replace it with this."
The pattern's constructor registers itself to match linalg::MatmulOp with a conversion cost of 1, telling MLIR's pattern matching system: "I know how to convert this operation."

The Main Transformation: `matchAndRewrite()`:

This is where the magic happens. When MLIR finds a matmul operation, it calls this method with the operation and its converted operands.
First, we extract the Matmul Details. The code extracts the inputs and outputs from the matmul operation. A matmul has two inputs (left and right matrices) and one output. The code validates this structure before proceeding.Then, the code checks that we're working with 2D float32 memrefs (memory references). If the matrices don't match this requirement, the conversion is skipped—this ensures we only convert operations we can safely handle.
We also need to differ between static and dynamic shapes. Matrices can have statically known dimensions or dynamically computed ones. The code handles both cases. For static dimensions (like a 128×128 matrix), it creates a constant. For dynamic dimensions, it uses memref::DimOp to extract the dimension at runtime and casts it to a 32-bit integer.

Now the code creates all the arguments that cblas_sgemm (single-precision general matrix multiply) needs:

- Order and transpose flags: Specifies row-major layout and no transposition
- Matrix dimensions: M (rows of A), N (columns of B), K (shared dimension)
- Leading dimensions: How memory is laid out for each matrix
- Alpha and beta scalars: The computation is C = alpha×A×B + beta×C
- Data pointers: Extracted from the converted LLVM memref structs

The code extracts pointers using LLVM::ExtractValueOp, which reaches into the LLVM struct representation of a memref to grab the raw pointer to the underlying data. Finally, all these arguments are bundled together and used to create an LLVM::CallOp that invokes the cblas_sgemm function. The original matmul operation is then erased, replaced by this function call.

The `getOrCreateSgemmFunc` helper method ensures the cblas_sgemm function is declared in the module. It first checks if the declaration already exists (to avoid duplicates). If not, it constructs the full function signature—specifying all 14 parameters and their types—and creates an LLVM function declaration.

The Pass's runOnOperation Method
This is where everything gets orchestrated:

1. Set up a conversion target: This tells MLIR which dialects are "legal" (can stay as-is) and which are "illegal" (must be converted). Here, Func, LLVM, MemRef, and Arith dialects are legal, but linalg.matmul operations are illegal.
2. Create a type converter: The LLVMTypeConverter knows how to transform MLIR types into their LLVM equivalents. For example, it converts high-level memref types into LLVM structs.
3. Register the pattern: The MatmulOpToBlasLibraryCall pattern is added to the pattern set with the type converter.
4. Apply the conversion: The applyPartialConversion function runs the pattern matching and rewriting process on the entire module. It returns success only if all illegal operations were successfully converted.

Putting It Together:

The whole flow is: scan the module for linalg.matmul operations → for each one, match it with our pattern → rewrite it as a cblas_sgemm call with proper dimension extraction and pointer marshaling → replace the original operation.
By the end, your high-level matrix multiplication operations have been transformed into direct calls to highly optimized BLAS library functions!