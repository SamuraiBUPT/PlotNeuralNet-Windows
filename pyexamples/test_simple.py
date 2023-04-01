import sys

sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_Conv(name="conv1", s_filer=512, n_filer=64, offset="(0,0,0)",
            to="(0,0,0)", height=64, depth=64, width=2, caption="Conv1"),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),

    to_Conv(name="conv2", s_filer=128, n_filer=64, offset="(1,0,0)",
            to="(pool1-east)", height=32, depth=32, width=2, caption="Conv2"),
    to_connection("pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),

    to_SoftMax("soft1", 10, "(3,0,0)", "(pool1-east)", caption="SOFT"),
    to_connection("pool2", "soft1"),
    to_Sum("sum1", offset="(1.5,0,0)", to="(soft1-east)", radius=2.5, opacity=0.6),
    to_connection("soft1", "sum1"),

    to_end()
]

arch2 = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_Conv(name="conv1", s_filer=64, n_filer=3, offset="(0,0,0)",
            to="(0,0,0)", height=64, depth=64, width=2, caption="Conv1"),
    to_Pool("pool1", offset="(1,0,0)", to="(conv1-east)",
            height=48, depth=48, width=2),
    # 先建立pool1，再进行连接
    to_connection("conv1", "pool1"),
    to_Conv(name="conv2", s_filer=128, n_filer=64, offset="(0,0,0)",
            to="(pool1-east)", height=48, depth=48, width=3, caption="Conv2"),

    to_Pool("pool2", offset="(1,0,0)", to="(conv2-east)",
            height=32, depth=32, width=4, caption="MaxPooling"),
    to_connection("conv2", "pool2"),

    to_Conv(name="conv3", s_filer=1, n_filer=128 * 8 * 8 * 8, offset="(2,0,0)",
            to="(pool2-east)", height=2, depth=2, width=10, caption="Flatten"),
    to_connection("pool2", "conv3"),

    # fc1
    to_SoftMax(name='fc1', s_filer=128 * 8 * 8 * 8, offset="(4,0,0)", to="(conv3-east)", width=1.5, height=1.5,
               depth=100, opacity=0.8, caption='FC1'),
    to_connection("conv3", "fc1"),
    # fc2
    to_SoftMax(name='fc2', s_filer=512, offset="(1.5,0,0)", to="(fc1-east)", width=1.5, height=1.5, depth=100,
               opacity=0.8, caption='FC2'),
    to_connection("fc1", "fc2"),
    # fc1
    to_SoftMax(name='fc3', s_filer=256, offset="(1.5,0,0)", to="(fc2-east)", width=1.5, height=1.5, depth=70,
               opacity=0.8, caption='FC3'),
    to_connection("fc2", "fc3"),
    to_end()
]

arch3 = [
    to_head('..'),
    to_cor(),
    to_begin(),
    to_ConvConvRelu("conv1", s_filer=256, n_filer=(0, 4), offset="(0,0,0)",
                    to="(0,0,0)", height=64, depth=64, width=(0, 2), caption="Conv1"),
    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch3, namefile + '.tex')


if __name__ == '__main__':
    main()
