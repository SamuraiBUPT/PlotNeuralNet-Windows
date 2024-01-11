import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, Callable, List
import os

class TorchPlot:
    def __init__(self, 
                 config: Dict):
        self.config = config
        
    def _forward_hook(self, layer_name: str, forward_layers, namelib) -> Callable:
        # define the register hook API function
        def hook(module, input, output, forward_layers=forward_layers, namelib=namelib):
            this_layer = {}
            print(f"Forward pass through layer: {layer_name} ({module.__class__.__name__})")
            this_layer['name'] = layer_name
            this_layer['type'] = module.__class__.__name__
            
            # process input
            for i, inp in enumerate(input, start=1):
                if isinstance(inp, torch.Tensor):
                    this_layer['input_shape'] = inp.shape
                elif isinstance(inp, (list, tuple)):
                    for j, item in enumerate(inp, start=1):
                        if isinstance(item, torch.Tensor):
                            this_layer['input_shape'] = item.shape
                        else:
                            print(f"  Item {j}: type {type(item).__name__}")
                else:
                    print(f"Input {i}: type {type(inp).__name__}")

            # process output
            if isinstance(output, (tuple, list)):
                for i, out in enumerate(output, start=1):
                    if isinstance(out, torch.Tensor):
                        this_layer['output_shape'] = out.shape
                    else:
                        print(f"Output {i}: type {type(out).__name__}")
            elif isinstance(output, torch.Tensor):
                this_layer['output_shape'] = output.shape
            else:
                print(f"Output: type {type(output).__name__}")
            print()
        
            # check before insert to avoid duplicate name
            for layer in forward_layers:
                if layer['name'] == layer_name:
                    if layer_name in namelib:
                        namelib[layer_name] += 1
                    else:
                        namelib[layer_name] = 1
                        
                    this_layer['name'] = f"{layer_name}_{namelib[layer_name]}"
            forward_layers.append(this_layer)
        
        return hook

    def _invoke_net_forward(self, net, input_tensor, forward_layers, namelib):
        for name, layer in net.named_children():
            if isinstance(layer, nn.Module):
                layer.register_forward_hook(self._forward_hook(name, forward_layers, namelib))
        with torch.no_grad():
            y = net(input_tensor)
        
        
    def analyze_net(self, 
                    net: nn.Module, 
                    input_tensor: torch.Tensor) -> List[Dict]:
        assert isinstance(net, nn.Module), "net must be a nn.Module"
        assert next(net.parameters()).device == input_tensor.device, "net and input_tensor must be on the same device"
        
        print("================ Analyzing ===================")
        net.eval()
        forward_layers: List[Dict] = [] # record the forward layers, each dict records the layer info
        namelib: Dict[str, int] = {}    # record the name of each layer as a counter, to avoid duplicate name
        
        self._invoke_net_forward(net, input_tensor, forward_layers, namelib)    # forward job
        
        for idx, item in enumerate(forward_layers):
            print(f"layer: {idx}")
            if 'output_shape' not in item or 'input_shape' not in item:
                raise Exception(f"layer {idx} has no output_shape/input_shape")
            else:
                for k, v in item.items():
                    print(k, ":", v)
                print()
                
        print("================= Finished ===================")
        # TODO: record the residual connection part, not implemented yet.
        
        """
            Note: It's actually not finished yet! We need to convert the `forward_layers` list of dicts 
            to the `arch`, for further generation.
            
            Now we temperarily return the `forward_layers`.
        """
        
        return forward_layers
    
    def generate(self, arch: List[str], filename: str):
        assert filename.endswith(".tex"), "filename must end with .tex"
        
        # first we generate the tex file
        with open(filename, "w") as f: 
            for c in arch:
                f.write(c)
        
        # then we will generate the pdf file
        os.system(f"pdflatex {filename}")
        os.system(f"rm {filename[:-4]}.aux {filename[:-4]}.log")
        