# TensorFlowToOnnx
python usage of tensorflow to onnx

This section includes
* **tfToOnnx:**
 I have trained a tf.keras model to predict handwritten digits using MNIST dataset for handwritten digits and coverted that model to .onnx format
* **saved_model.onnx:**
 onnx model generated from previous step
* **C_Api_Sample1.cpp:**
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
My concern is `None` dimention of the model is being converted to `-1` Do we need to fix it ? because in the next stage of **C_Api_Sample1.cpp:** for the line https://github.com/UTA-HEP-Computing/TensorFlowToOnnx/blob/master/C_Api_Sample1.cpp#L156
```
CHECK_STATUS(OrtCreateTensorWithDataAsOrtValue(allocator_info, input_tensor_values.data(), input_tensor_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
```
I get 

```
size overflow
```
