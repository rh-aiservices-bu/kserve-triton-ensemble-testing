# Ensemble Models Example with Triton and Kserve

## Deployment

- The [Ensemble](https://developer.nvidia.com/blog/serving-ml-model-pipelines-on-nvidia-triton-inference-server-with-ensemble-models/) model is available in the `model01` folder.
- The whole content of the model folder must be uploaded to an object store bucket.
- In OpenShift AI, create a Data Connection pointing to this bucket.
- Import the custom runtime for Triton in OpenShift AI, as a Single Model Serving runtime. Two different runtimes are provided in the `runtime` folder, depending if you want to use the gRPC or REST API.
- Serve the model in OpenShift AI using the custom runtime you imported, pointing it to the data connection.

## Usage

Two example notebooks are provided as examples to connect to the model using either gRPC or REST.

Please note that this model is just a stub that does mostly nothing except going trough different steps to illustrate how Ensemble in Triton works.
