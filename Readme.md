# Ensemble Models with Triton and KServe

An ensemble model represents a pipeline of one or more models and the connection of input and output tensors between those models. Ensemble models are intended to be used to encapsulate a procedure that involves multiple models, such as “data preprocessing -> inference -> data postprocessing”. Using ensemble models for this purpose can avoid the overhead of transferring intermediate tensors and minimize the number of requests that must be sent to Triton. [Ref](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1150/user-guide/docs/models_and_schedulers.html#ensemble-models).

This is a simple example on how to deploy and use Ensemble models in OpenShift AI with the Triton runtime.

## Requirements

- Triton must be deployed as a custom Single Model Serving Runtime in OpenShift AI. Two examples are provided (you can deploy both if you want to test different configurations):
  - A REST interface version: [runtime-rest.yaml](runtime/runtime-rest.yaml)
  - A gRPC interface version: [runtime-grpc.yaml](runtime/runtime-grpc.yaml)
- An Ensemble model. An example is available in the [model01](model01) folder. Please note that this model is just a stub that does mostly nothing except going trough different steps to illustrate how Ensemble works in Triton.

## Deployment

- Copy the whole content of the model folder (so normally multiple models, plus the Ensemble definition) to an object store bucket.
- In OpenShift AI, create a Data Connection pointing to the bucket.
- Serve the model in OpenShift AI using the custom runtime you imported, pointing it to the data connection.
- After a few seconds/minutes, the model is served and an inference endpoint is available.

## Usage

Two example notebooks, `test-ensemble-rest.ipynb` and `test-ensemble-grpc.ipynb` are provided as examples to connect to the model using either gRPC or REST.

Please note that this model is just a stub that does mostly nothing except going trough different steps to illustrate how Ensemble works in Triton.
