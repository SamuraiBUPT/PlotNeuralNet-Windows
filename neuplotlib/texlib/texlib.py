import os

"""
    This file stores the functions to generate the tex file for the network architecture,
    as a library for the top-level wrapper.
    
    Normally, we avoid to use the library directly, but use the wrapper instead.
"""

current_path = os.path.split(os.path.realpath(__file__))[0]

def to_head() -> str:
    layers_path = os.path.join(current_path, 'layers/' ).replace('\\', '/')
    return r"""
\documentclass[border=8pt, multi, tikz]{standalone} 
\usepackage{import}
\subimport{"""+ layers_path + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image 
"""

def to_color() -> str:
    return r"""
\def\ConvColor{rgb:yellow, 5; red, 2.5; white, 5}
\def\ConvReluColor{rgb:yellow, 5; red, 5; white,5}
\def\PoolColor{rgb:red, 1; black, 0.3}
\def\UnpoolColor{rgb:blue, 2; green, 1; black, 0.3}
\def\FcColor{rgb:blue, 5; red, 2.5; white, 5}
\def\FcReluColor{rgb:blue, 5; red, 5; white, 4}
\def\SoftmaxColor{rgb:magenta, 5; black, 7}   
\def\SumColor{rgb:blue, 5; green, 15}
"""

def to_begin() -> str:
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""

# layers definition
def to_input(pathfile, 
             to='(-3,0,0)', 
             width=8, 
             height=8, 
             name="temp") -> str:
    return r"""
\node[canvas is zy plane at x=0] (""" + name + """) at """+ to +""" {\includegraphics[width="""+ str(width)+"cm"+""",height="""+ str(height)+"cm"+"""]{"""+ pathfile +"""}};
"""

# Conv
def to_Conv(name, 
            z_label=256, x_label=64, 
            base="(0,0,0)", offset="(0,0,0)", 
            width=1, height=40, depth=40, 
            caption=" ") -> str:
    return r"""
\pic[shift={"""+ offset +"""}] at """+ base +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(x_label) +""", }},
        zlabel="""+ str(z_label) +r""",
        fill=\ConvColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""


# Conv,Conv,relu
# Bottleneck
def to_ConvConvRelu(name, 
                    z_label=256, x_label=(64,64), 
                    base="(0,0,0)", offset="(0,0,0)", 
                    width=(2,2), height=40, depth=40, 
                    caption=" ") -> str:
    return r"""
\pic[shift={ """+ offset +""" }] at """+ base +""" 
    {RightBandedBox={
        name="""+ name +""",
        caption="""+ caption +""",
        xlabel={{ """+ str(x_label[0]) +""", """+ str(x_label[1]) +""" }},
        zlabel="""+ str(z_label) +""",
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +""",
        width={ """+ str(width[0]) +""" , """+ str(width[1]) +""" },
        depth="""+ str(depth) +"""
        }
    };
"""


# Pool
def to_Pool(name, 
            base="(0,0,0)", offset="(0,0,0)", 
            width=1, height=32, depth=32, 
            opacity=0.5, 
            caption=" ") -> str:
    return r"""
\pic[shift={ """+ offset +""" }] at """+ base +""" 
    {Box={
        name="""+name+""",
        caption="""+ caption +r""",
        fill=\PoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# unpool4, 
def to_UnPool(name, 
              base="(0,0,0)", offset="(0,0,0)", 
              width=1, 
              height=32, 
              depth=32, 
              opacity=0.5, 
              caption=" ") -> str:
    return r"""
\pic[shift={ """+ offset +""" }] at """+ base +""" 
    {Box={
        name="""+ name +r""",
        caption="""+ caption +r""",
        fill=\UnpoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""


def to_ConvRes(name, 
               x_label=64, z_label=256, 
               base="(0,0,0)", offset="(0,0,0)", 
               width=6, height=40, depth=40, 
               opacity=0.2, 
               caption=" " ) -> str:
    return r"""
\pic[shift={ """+ offset +""" }] at """+ base +""" 
    {RightBandedBox={
        name="""+ name + """,
        caption="""+ caption + """,
        xlabel={{ """+ str(x_label) + """, }},
        zlabel="""+ str(z_label) +r""",
        fill={rgb:white,1;black,3},
        bandfill={rgb:white,1;black,2},
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""


# ConvSoftMax
def to_ConvSoftMax(name, 
                   z_label=40, 
                   base="(0,0,0)", offset="(0,0,0)", 
                   width=1, height=40, depth=40, 
                   caption=" ") -> str:
    return r"""
\pic[shift={"""+ offset +"""}] at """+ base +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        zlabel="""+ str(z_label) +""",
        fill=\SoftmaxColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# SoftMax
def to_SoftMax(name, 
               z_label=10, 
               base="(0,0,0)", offset="(0,0,0)", 
               width=1.5, height=3, depth=25, 
               opacity=0.8, 
               caption=" " ) -> str:
    return r"""
\pic[shift={"""+ offset +"""}] at """+ base +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        xlabel={{" ","dummy"}},
        zlabel="""+ str(z_label) +""",
        fill=\SoftmaxColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Sum(name, 
           base="(0,0,0)", offset="(0,0,0)", 
           radius=2.5, 
           opacity=0.6) -> str:
    return r"""
\pic[shift={"""+ offset +"""}] at """+ base +""" 
    {Ball={
        name=""" + name +""",
        fill=\SumColor,
        opacity="""+ str(opacity) +""",
        radius="""+ str(radius) +""",
        logo=$+$
        }
    };
"""


def to_connection(src, dest) -> str:
    return r"""
\draw [connection]  ("""+src+"""-east)    -- node {\midarrow} ("""+dest+"""-west);
"""


def to_skip(src, dest, pos=1.25) -> str:
    return r"""
\path ("""+ src +"""-southeast) -- ("""+ src +"""-northeast) coordinate[pos="""+ str(pos) +"""] ("""+ src +"""-top) ;
\path ("""+ dest +"""-south)  -- ("""+ dest +"""-north)  coordinate[pos="""+ str(pos) +"""] ("""+ dest +"""-top) ;
\draw [copyconnection]  ("""+src+"""-northeast)  
-- node {\copymidarrow}("""+src+"""-top)
-- node {\copymidarrow}("""+dest+"""-top)
-- node {\copymidarrow} ("""+dest+"""-north);
"""


def to_end() -> str:
    return r"""
\end{tikzpicture}
\end{document}
"""

torch_to_texlib = {
    "Conv2d": to_Conv,
    "MaxPool2d": to_Pool,
    "Linear": to_SoftMax
}