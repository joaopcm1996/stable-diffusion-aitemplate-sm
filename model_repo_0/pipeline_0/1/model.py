import torch
import sys, os
import time
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack, from_dlpack

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.import_path import import_parent
from diffusers import DPMSolverMultistepScheduler


class TritonPythonModel:

    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]),
                                               "generated_image")["data_type"])
        
        self.model_dir = args['model_repository']
        
        sys.path.append(self.model_dir)
        from src import pipeline_stable_diffusion_ait
        from src.pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
        
        self.local_dir = f'{self.model_dir}/tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2-1-base'
        
        os.environ["AIT_MODEL_PATH"] = f'{self.model_dir}/tmp/'
        self.pipe = StableDiffusionAITPipeline.from_pretrained(
                                                        self.local_dir,
                                                        revision="fp16",
                                                        torch_dtype=torch.float16,
                                                        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        
    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "prompt")
            input_text = inp.as_numpy()[0][0].decode()

            with torch.autocast("cuda"):
                image_array = self.pipe(input_text, 512, 512,output_type='numpy').images

            decoded_image = (image_array * 255).round().astype("uint8")
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    decoded_image
                )
            ])
            
            responses.append(inference_response)
            
        return responses
