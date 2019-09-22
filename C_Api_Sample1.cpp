// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

//*****************************************************************************
// helper function to check for status
#define CHECK_STATUS(expr)                               \
  {                                                      \
    OrtStatus* onnx_status = (expr);                     \
    if (onnx_status != NULL) {                           \
      const char* msg = OrtGetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                      \
      OrtReleaseStatus(onnx_status);                     \
      exit(1);                                           \
    }                                                    \
  }

int main(int argc, char* argv[]) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  OrtEnv* env;
  CHECK_STATUS(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  // initialize session options if needed
  OrtSessionOptions* session_options;
  CHECK_STATUS(OrtCreateSessionOptions(&session_options));
  OrtSetSessionThreadPoolSize(session_options, 1);

  // Sets graph optimization level
  OrtSetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

  // Optionally add more execution providers via session_options
  // E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
  // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

  //*************************************************************************
  // create session and load model into memory

  OrtSession* session;
  const char* model_path = "saved_model.onnx";

  printf("Using Onnxruntime C API\n");
  CHECK_STATUS(OrtCreateSession(env, model_path, session_options, &session));

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  size_t num_input_nodes;
  size_t num_output_nodes;
  OrtStatus* status_In;
  OrtStatus* status_Out;
  OrtAllocator* allocator;
  CHECK_STATUS(OrtGetAllocatorWithDefaultOptions(&allocator));

  // print number of model input nodes
  status_In = OrtSessionGetInputCount(session, &num_input_nodes);
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  

  printf("Number of inputs = %zu\n", num_input_nodes);

  // print number of model output nodes
  status_Out = OrtSessionGetOutputCount(session, &num_output_nodes);
  std::vector<const char*> output_node_names(num_output_nodes);
  std::vector<int64_t> output_node_dims;  

  printf("Number of outputs = %zu\n", num_output_nodes);
 
  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name;
    status_In = OrtSessionGetInputName(session, i, allocator, &input_name);
    printf("Input %zu : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    OrtTypeInfo* typeinfo;
    status_In = OrtSessionGetInputTypeInfo(session, i, &typeinfo);
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CHECK_STATUS(OrtCastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
    CHECK_STATUS(OrtGetTensorElementType(tensor_info, &type));
    printf("Input %zu : type=%d\n", i, type);

    // print input shapes/dims
    size_t num_dims;
    CHECK_STATUS(OrtGetDimensionsCount(tensor_info, &num_dims));
    printf("Input %zu : num_dims=%zu\n", i, num_dims);
    input_node_dims.resize(num_dims);
    OrtGetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
    for (size_t j = 0; j < num_dims; j++)
      printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);

    OrtReleaseTypeInfo(typeinfo);
  }

  // iterate over all output nodes
  for (size_t i = 0; i < num_output_nodes; i++) {
    // print output node names
    char* output_name;
    status_Out = OrtSessionGetOutputName(session, i, allocator, &output_name);
    printf("Output %zu : name=%s\n", i, output_name);
    output_node_names[i] = output_name;

    // print output node types
    OrtTypeInfo* typeinfo;
    status_Out = OrtSessionGetOutputTypeInfo(session, i, &typeinfo);
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CHECK_STATUS(OrtCastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
    CHECK_STATUS(OrtGetTensorElementType(tensor_info, &type));
    printf("Output %zu : type=%d\n", i, type);

    // print output shapes/dims
    size_t num_dims;
    CHECK_STATUS(OrtGetDimensionsCount(tensor_info, &num_dims));
    printf("Output %zu : num_dims=%zu\n", i, num_dims);
    output_node_dims.resize(num_dims);
    OrtGetDimensions(tensor_info, (int64_t*)output_node_dims.data(), num_dims);
    for (size_t j = 0; j < num_dims; j++)
      printf("Output %zu : dim %zu=%jd\n", i, j, output_node_dims[j]);

    OrtReleaseTypeInfo(typeinfo);
  }

  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  size_t input_tensor_size = 28 * 28 * 1;
  std::vector<float> input_tensor_values(input_tensor_size);
  //std::vector<const char*> output_node_names = {"softmaxout_1"};

  // initialize input data with values in [0.0, 1.0]
  for (size_t i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);
  printf("DeboCheck1 \n");  

  // create input tensor object from data values
  OrtAllocatorInfo* allocator_info;
  CHECK_STATUS(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info));
  printf("DeboCheck2 \n");
  OrtValue* input_tensor = NULL;
  printf("DeboCheck3 \n");
  CHECK_STATUS(OrtCreateTensorWithDataAsOrtValue(allocator_info, input_tensor_values.data(), input_tensor_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
  printf("DeboCheck4 \n");
  int is_tensor;
  CHECK_STATUS(OrtIsTensor(input_tensor, &is_tensor));
  printf("DeboCheck5 \n");
  assert(is_tensor);
  OrtReleaseAllocatorInfo(allocator_info);
  printf("DeboCheck6 \n"); 
  // score model & input tensor, get back output tensor
  OrtValue* output_tensor = NULL;
  CHECK_STATUS(OrtRun(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor));
  CHECK_STATUS(OrtIsTensor(output_tensor, &is_tensor));
  assert(is_tensor);
  printf("DeboCheck7 \n"); 
  // Get pointer to output tensor float values
  float* floatarr;
  CHECK_STATUS(OrtGetTensorMutableData(output_tensor, (void**)&floatarr));
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores 
  for (int i = 0; i < 10; i++)
    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

  OrtReleaseValue(output_tensor);
  OrtReleaseValue(input_tensor);
  OrtReleaseSession(session);
  OrtReleaseSessionOptions(session_options);
  OrtReleaseEnv(env);
  printf("Done!\n");
  return 0;
}
