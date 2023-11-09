import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, Callable

class TorchPlot:
    def __init__(self, 
                 config: Dict):
        self.config = config
        self.forward_layers = []
        self.namelib: Dict[str, int] = {}
        
    def _forward_hook(self, layer_name: str) -> Callable:
          
        # define the register hook API function
        def hook(module, input, output):
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
            for layer in self.forward_layers:
                if layer['name'] == layer_name:
                    if layer_name in self.namelib:
                        self.namelib[layer_name] += 1
                    else:
                        self.namelib[layer_name] = 1
                        
                    this_layer['name'] = f"{layer_name}_{self.namelib[layer_name]}"
            self.forward_layers.append(this_layer)
        
        return hook

    def _invoke_net_forward(self, net, input_tensor):
        self.forward_layers.clear()
        self.namelib.clear()
        
        for name, layer in net.named_children():
            if isinstance(layer, nn.Module):
                layer.register_forward_hook(self._forward_hook(name))
        y = net(input_tensor)
        
        
    def plot(self, net, input_tensor):
        self._invoke_net_forward(net, input_tensor)
        
        # TODO: finish the plot function with latex, now we just print the forward pass
        for idx, item in enumerate(self.forward_layers):
            print(f"layer: {idx}")
            if 'output_shape' not in item or 'input_shape' not in item:
                raise Exception(f"layer {idx} has no output_shape/input_shape")
            else:
                for k, v in item.items():
                    print(k, ":", v)
                print()