# 一、PlotNeuralNet简介

## 1. Introduction

**PlotNeuralNet**库是用于绘制神经网络的工具库，其绘制的神经网络较为干净整洁，比较适合用于**科研论文写作**等工作中，在此笔者整理了有关该库的使用方法，希望更多朋友能够借助这个库绘制出更多优美的神经网络。

以下是一些通过该库的代码绘制的Neural Netowrk Figures:

<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308846-c2231880-049c-11e9-8763-3daa1024de78.png"></p>

$$FCN-8$$

<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308873-e2eb6e00-049c-11e9-9587-9da6bdec011b.png"></p>

$$FCN-32$$

<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308911-03b3c380-049d-11e9-92d9-ce15669017ad.png"></p>

$$Holistically-Nested Edge Detection$$

## 2. PlotNeuralNet组成和原理

### 2.1 代码组成
该仓库的代码借助三种语言进行绘制：

+ `python` 用于进行设计您自己的神经网络，作为主要的组织脚本文件
+ `shell` 用于组织python和latex的运行，作为中间件。（在Linux系统上较为常见和通用）
+ `LaTex` 用于绘制神经网络，通过python进行调用

### 2.2 运行原理
项目通过python调用latex代码进行绘制，这里可以给出一份简短的例子了解工作原理：

```python
# Conv
def to_Conv( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +""",
        fill=\ConvColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""
```

$$pycore.tikzeng.py$$

其实从这里就可以看出，通过设定函数参数、返回一个latex的字符串，相当于调用了latex进行绘制。（最后有进行组织绘制的代码）

### 2.3 Linux友好，Windows有一定问题

不过，值得注意的是，**原仓库设计是针对Linux用户的解决方案，在Ubuntu系统下能够顺利运行，在Windows系统上运行的时候，可能存在一定问题。**

因此笔者本人对该仓库进行了fork并且做出一定修改，使其能够在windows系统上运行。

## 3. forked Github Repository

源仓库是面向Linux系统的神经网络可视化工具，在windows上运行可能会有一定的错误信息。

下面是笔者修改过后的代码仓库地址：（fork自原仓库）

[SamuraiBUPT: PlotNeuralNet-Windows](https://github.com/SamuraiBUPT/PlotNeuralNet-Windows)


# 二、快速开始-Windows

下面是针对windows用户的启动方案

## 1. 前置设置

+ 确保您的电脑上安装了LaTex。官方仓库推荐安装 [MikTeX](https://miktex.org/download) ，不过本人仅仅安装了texlive就可以正常运行~
+ 确保您的Windows操作系统具有运行Shell脚本的环境。推荐使用[Git bash](https://git-scm.com/download/win) 或者 [Cygwin](https://www.cygwin.com/)

## 2. 克隆本仓库

您可以使用 `git clone https://github.com/SamuraiBUPT/PlotNeuralNet-Windows.git` 来完成仓库的克隆操作。

## 3. 跟随以下示例了解如何运行~
+ 将克隆后的项目文件夹用IDE打开
+ 启用bash环境（如果您已经有Git Bash，并且配置好了环境变量，您可以直接在终端输入`bash` 或者`sh` 进入shell脚本的运行环境。您会看见终端前方的指示符从windows系统的黑白变成了其他颜色。
+ 进入`pyexamples`路径，（`cd pyexamples/`）
+ 输入`bash ../tikzmake.sh test_simple` 命令来查看您的第一份神经网络图的绘制结果。

## 4. 解释
如您所见，当您在pyexamples路径下创建好自己的python脚本后，仅需要在该路径下调用上一层路径的tikzmake.sh脚本来运行即可。

`bash`指令是在指定Shell脚本的编译器

`../tikzmake.sh`是在指定：执行上一层目录下的`tikzmake.sh`脚本

`test_simple`是当前路径下的一份python文件的文件名，**注意不需要加上`.py`的后缀！**。在这里，这个文件名会作为shell脚本的参数传入，决定了shell脚本执行的目标是哪份文件。


# 三、用法

了解如何上手该项目代码后，我们可以开始着手绘制自己的神经网络：

## 1. 常规的初始化

**一份常规的组织神经网络的python代码，应该有如下骨架**：

```python
import sys

sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # your architecture here...
    # ...
    # ...
    # ...

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
```

如您所见，其实唯一需要您进行改动的，是`arch` 列表里面的内容，在`to_begin()` 和 `to_end()` 函数之间的部分，在该部分进行您自己的神经网络设计。

# 四、函数描述
有关函数的描述您可以参阅这篇[博客](https://blog.csdn.net/qq_40876059/article/details/124442944)