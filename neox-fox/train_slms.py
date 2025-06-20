# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from megatron.neox_arguments import NeoXArgs
from megatron.training_slms import pretrain

if __name__ == "__main__":
    seeds = [42, 123, 2024]
    results = []

    for seed in seeds:
        neox_args = NeoXArgs.from_ymls(["/content/gpt-neox-baseline/configs/pythia-2-4-256-1k_fox.yml"])
        neox_args.seed = seed
        neox_args.configure_distributed_args()
        neox_args.build_tokenizer()
        neox_args.initialize_tensorboard_writer()

        result = pretrain(neox_args=neox_args)  
        results.append({'seed': seed, 'result': result})

    import json
    with open('training_results_fox.json', 'w') as f:
        json.dump(results, f, indent=2)
    
