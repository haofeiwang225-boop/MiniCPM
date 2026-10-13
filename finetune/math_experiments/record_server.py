"""Run under the installed vLLM environment; snapshot resolved API sampling defaults."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import vllm
try:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
except ModuleNotFoundError as exc:
    if exc.name != "vllm.entrypoints.openai.chat_completion":
        raise
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

p=argparse.ArgumentParser()
p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--port',type=int,default=8000);p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
request=ChatCompletionRequest(model='math-model',messages=[dict(role='user',content='audit')],
    temperature=1.,top_p=.95,n=1,chat_template_kwargs=dict(enable_thinking=True))
resolved=request.to_sampling_params(get_max_tokens(131072,None,100,{}),{})
keys=['temperature','top_p','top_k','min_p','repetition_penalty','presence_penalty','frequency_penalty',
      'seed','n','stop','stop_token_ids','ignore_eos','max_tokens']
record=dict(model=str(Path(a.model).resolve()),served_model_name='math-model',seed=a.seed,port=a.port,
    vllm_version=vllm.__version__,generation_config='vllm',max_model_len=131072,gpu_memory_utilization=.65,
    max_num_seqs=1,request_seed=None,example_input_tokens=100,
    resolved_sampling_for_example={k:getattr(resolved,k,None) for k in keys},
    max_tokens_note='derived remaining context, not a client generation cap',
    source_sha256={str(Path(inspect.getfile(obj))):hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
                   for obj in (ChatCompletionRequest,get_max_tokens)},
    nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True))
with a.output.open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record,indent=2))
