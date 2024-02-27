import numpy as np
import sys
import json
import io

import triton_python_backend_utils as pb_utils

import os
import subprocess

from datetime import datetime

# import redis


class TritonPythonModel:

    def initialize(self, args):
        
        
        # self.redis_db = redis.Redis(host='redis-service', port=6379, db=0, decode_responses=True)

        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0").as_numpy()
            
            card_number = 123456789
            
            feature_vector_list = ['12','23','45'] #eval(self.redis_db.get(card_number))
            
            full_feature_list = in_0.tolist()[0] + feature_vector_list

            feature_merger_output = np.asarray([full_feature_list]).astype(np.float32)

            out_tensor_0 = pb_utils.Tensor("OUTPUT_0", feature_merger_output)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

            print(responses)
            print(out_tensor_0)
            
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
