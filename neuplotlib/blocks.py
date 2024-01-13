from .texlib import *

#define new block
def block_2ConvPool(name, 
                    botton, 
                    top, 
                    z_label=256, 
                    x_label=64, 
                    offset="(1,0,0)", 
                    size=(32,32,3.5), 
                    opacity=0.5):
    return [
        to_ConvConvRelu( 
            name="ccr_{}".format( name ),
            z_label=str(z_label), 
            x_label=(x_label,x_label), 
            base="({}-east)".format( botton ), 
            offset=offset, 
            width=(size[2],size[2]), 
            height=size[0], 
            depth=size[1],   
            ),    
        to_Pool(         
            name="{}".format( top ), 
            base="(ccr_{}-east)".format( name ),  
            offset="(0,0,0)", 
            width=1,         
            height=size[0] - int(size[0]/4), 
            depth=size[1] - int(size[0]/4), 
            opacity=opacity, ),
        to_connection( 
            "{}".format( botton ), 
            "ccr_{}".format( name )
            )
    ]


def block_Unconv(name, 
                 botton, 
                 top, 
                 z_label=256, 
                 x_label=64, 
                 offset="(1,0,0)", 
                 size=(32,32,3.5), 
                 opacity=0.5):
    return [
        to_UnPool(  name='unpool_{}'.format(name),    offset=offset,    base="({}-east)".format(botton),         width=1,              height=size[0],       depth=size[1], opacity=opacity ),
        to_ConvRes( name='ccr_res_{}'.format(name),   offset="(0,0,0)", base="(unpool_{}-east)".format(name),    z_label=str(z_label), x_label=str(x_label), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='ccr_{}'.format(name),       offset="(0,0,0)", base="(ccr_res_{}-east)".format(name),   z_label=str(z_label), x_label=str(x_label), width=size[2], height=size[0], depth=size[1] ),
        to_ConvRes( name='ccr_res_c_{}'.format(name), offset="(0,0,0)", base="(ccr_{}-east)".format(name),       z_label=str(z_label), x_label=str(x_label), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='{}'.format(top),            offset="(0,0,0)", base="(ccr_res_c_{}-east)".format(name), z_label=str(z_label), x_label=str(x_label), width=size[2], height=size[0], depth=size[1] ),
        to_connection( 
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
            )
    ]


def block_Res(num, 
              name, 
              botton, 
              top, 
              z_label=256, 
              x_label=64, 
              offset="(0,0,0)", 
              size=(32,32,3.5), 
              opacity=0.5):
    lys = []
    layers = [ *[ '{}_{}'.format(name,i) for i in range(num-1) ], top]
    for name in layers:        
        ly = [to_Conv( 
                name='{}'.format(name),       
                base="({}-east)".format( botton ),   
                offset=offset, 
                z_label=str(z_label), 
                x_label=str(x_label), 
                width=size[2],
                height=size[0],
                depth=size[1]),
            to_connection( 
                "{}".format( botton  ), 
                "{}".format( name ))
            ]
        botton = name
        lys+=ly
    
    lys += [
        to_skip(src=layers[1], dest=layers[-2], pos=1.25),
    ]
    return lys


