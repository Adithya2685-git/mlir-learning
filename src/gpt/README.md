MLIR pass works, i.e. model can be lowered with MLIR is available as object code.

TODO: The following arguments are not used in the MLIR code and thus can be evicted. But so far, I couldn't find a pass that cleans up those dead/unused arguments.

```
%arg0: tensor<1x1x1024x1024xi1>, %arg1: tensor<f32>, %arg2: tensor<1x1x1024x1024xi1>, %arg3: tensor<f32>, %arg4: tensor<1x1x1024x1024xi1>, %arg5: tensor<f32>, %arg6: tensor<1x1x1024x1024xi1>, %arg7: tensor<f32>, %arg8: tensor<1x1x1024x1024xi1>, %arg9: tensor<f32>, %arg10: tensor<1x1x1024x1024xi1>, %arg11: tensor<f32>, %arg12: tensor<1x1x1024x1024xi1>, %arg13: tensor<f32>, %arg14: tensor<1x1x1024x1024xi1>, %arg15: tensor<f32>, %arg16: tensor<1x1x1024x1024xi1>, %arg17: tensor<f32>, %arg18: tensor<1x1x1024x1024xi1>, %arg19: tensor<f32>, %arg20: tensor<1x1x1024x1024xi1>, %arg21: tensor<f32>, %arg22: tensor<1x1x1024x1024xi1>, %arg23: tensor<f32>
```