# TensorFlowToOnnx
python usage of tensorflow to onnx

This section includes
* **tfToOnnx:**
 I have trained a tf.keras model to predict handwritten digits using MNIST dataset for handwritten digits and coverted that model to .onnx format
* **saved_model.onnx:**
 onnx model generated from previous step
* **C_Api_Sample4.cpp:**
 c/c++ API of onnx_runtime to use onnx model for scoring
 
 The **saved_model.onnx:** model has following dimention
```
Input:
input name flatten_1_input:0
input shape [None, 28, 28]
input type tensor(float)

Output:
output_name dense_3/Softmax:0
output shape [None, 10]
output type tensor(float)
```
The same **saved_model.onnx:** has been loaded to **C_Api_Sample1.cpp:** using onnx_runtime's c/c++ API and I get following shapes
```
Using Onnxruntime C API
Number of inputs = 1
Number of outputs = 1
Input 0 : name=flatten_1_input:0
Input 0 : type=1
Input 0 : num_dims=3
Input 0 : dim 0=-1
Input 0 : dim 1=28
Input 0 : dim 2=28
Output 0 : name=dense_3/Softmax:0
Output 0 : type=1
Output 0 : num_dims=2
Output 0 : dim 0=-1
Output 0 : dim 1=10
```
I faced an issue while preparing input data with above dimension of .onnx model. Since `Input 0 : dim 0=-1` provides an invalid shape for input tensor but this is now resolved with useful suggestions from onnx developers. So now with onnx_runtime c/c++ API, I can predict MNIST handwritten digits with a .onnx inference. The output of C_Api_Sample4.cpp file is as follows
```
Using Onnxruntime C API
Number of inputs = 1
Number of outputs = 1
Input 0 : name=flatten_input:0
Input 0 : type=1
Input 0 : num_dims=3
Input 0 : dim 0=1
Input 0 : dim 1=28
Input 0 : dim 2=28
Output 0 : name=dense_1/Softmax:0
Output 0 : type=1
Output 0 : num_dims=2
Output 0 : dim 0=1
Output 0 : dim 1=10
Score for class [0] =  0.000000
Score for class [1] =  0.000000
Score for class [2] =  0.000000
Score for class [3] =  0.000000
Score for class **[4] =  0.996822**
Score for class [5] =  0.000003
Score for class [6] =  0.000000
Score for class [7] =  0.000277
Score for class [8] =  0.001590
Score for class [9] =  0.001308
Label for the input test data  =  **4**
Done!
```
