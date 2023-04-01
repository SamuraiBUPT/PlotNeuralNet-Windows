# PlotNeuralNet - Zh_CN 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2526396.svg)](https://doi.org/10.5281/zenodo.2526396)

该仓库是PlotNeuralNet的中文仓库。

**PlotNeuralNet**通过LaTex、Python、Shell三种方式辅助绘制神经网络。您可以查看示例以了解它们是如何制作的。此外，让我们整合您所做的任何改进并修复任何错误，以帮助更多的人使用此代码。

+ `python`脚本用于进行个人神经网络的搭建
+ `shell`脚本作为中间件，串联python与LaTex的协作工作
+  `LaTex`用于绘制优美的神经网络结构图。

## 神经网络示例图

下面是一些用**PlotNeuralNet**绘制的神经网络结构图的示例：

<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308846-c2231880-049c-11e9-8763-3daa1024de78.png" width="85%" height="85%"></p>
<h6 align="center">FCN-8 (<a href="https://www.overleaf.com/read/kkqntfxnvbsk">view on Overleaf</a>)</h6>


<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308873-e2eb6e00-049c-11e9-9587-9da6bdec011b.png" width="85%" height="85%"></p>
<h6 align="center">FCN-32 (<a href="https://www.overleaf.com/read/wsxpmkqvjnbs">view on Overleaf</a>)</h6>


<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308911-03b3c380-049d-11e9-92d9-ce15669017ad.png" width="85%" height="85%"></p>
<h6 align="center">Holistically-Nested Edge Detection (<a href="https://www.overleaf.com/read/jxhnkcnwhfxp">view on Overleaf</a>)</h6>

## 快速开始-Windows

下面是针对windows用户的启动方案

### 1. 前置设置
+ 确保您的电脑上安装了LaTex。官方仓库推荐安装 [MikTeX](https://miktex.org/download) ，不过本人仅仅安装了texlive就可以正常运行~
+ 确保您的Windows操作系统具有运行Shell脚本的环境。推荐使用[Git bash](https://git-scm.com/download/win) 或者 [Cygwin](https://www.cygwin.com/)

### 2. 克隆本仓库
您可以使用 `git clone https://github.com/SamuraiBUPT/PlotNeuralNet-Windows.git` 来完成仓库的克隆操作。

### 3. 跟随以下示例了解如何运行~
+ 将克隆后的项目文件夹用IDE打开
+ 启用bash环境（如果您已经有Git Bash，并且配置好了环境变量，您可以直接在终端输入`bash` 或者`sh` 进入shell脚本的运行环境。您会看见终端前方的指示符从windows系统的黑白变成了其他颜色。
+ 进入`pyexamples`路径，（`cd pyexamples/`）
+ 输入`bash ../tikzmake.sh test_simple` 命令来查看您的第一份神经网络图的绘制结果。

## 解释
如您所见，当您在pyexamples路径下创建好自己的python脚本后，仅需要在该路径下调用上一层路径的tikzmake.sh脚本来运行即可。

`bash`指令是在指定Shell脚本的编译器

`../tikzmake.sh`是在指定：执行上一层目录下的`tikzmake.sh`脚本

`test_simple`是当前路径下的一份python文件的文件名，**注意不需要加上`.py`的后缀！**。在这里，这个文件名会作为shell脚本的参数传入，决定了shell脚本执行的目标是哪份文件。


# 用法
## 常规的初始化
一份常规的组织神经网络的python代码，应该有如下骨架：

```angular2html
import sys

sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    // your architecture here...

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
```

如您所见，其实唯一需要您进行改动的，是`arch` 列表里面的内容，在`to_begin()` 和 `to_end()` 函数之间的部分。
