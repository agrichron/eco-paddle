import paddle.fluid as fluid
from pathlib import Path
import numpy as np
import os
import shutil
import pickle

from visualdl import LogWriter

from opts import parser
from model import ECOfull
from config import parse_config, print_configs
from reader import KineticsReader


args = parser.parse_args()

cfg = parse_config('config.txt')
# print_configs(cfg, 'TRAIN')
main_program = fluid.default_main_program()
start_program = fluid.default_startup_program()
place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
with fluid.program_guard(main_program, start_program):
    
    # data placeholder
    input = fluid.data(name='data', shape=[-1, 3, 224, 224], dtype='float32')
    label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
    print(f'label shape:{label.shape}')
    model = ECOfull(input, num_segments= args.num_segments)
    net_out = model()
    
    cost = fluid.layers.softmax_with_cross_entropy(net_out, label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(net_out, label)
    # test program
    eval_program = main_program.clone(for_test=True)
    # optimizer
    fluid.optimizer.SGD(args.lr).minimize(avg_cost)

 # 验证集
val_reader = KineticsReader('eco', 'valid', cfg).create_reader()
feeder = fluid.DataFeeder([input, label], place)
# 初始化参数
exe = fluid.Executor(place=place)
exe.run(start_program)
val_exe = fluid.Executor(place=place)
fluid.io.load(main_program, 'models/eco_full_best', val_exe)
val_avg_loss = []
val_acc = []
fetch_list = [avg_cost.name, acc.name]
for data in val_reader():
    avg_cost_, acc_ = exe.run(eval_program, feed=feeder.feed(data), fetch_list=fetch_list)
    val_avg_loss += [avg_cost_]
    val_acc += [acc_]
val_avg_loss = np.array(val_avg_loss).mean()
val_avg_acc = np.array(val_acc).mean()
print(f'val avg_loss:{val_avg_loss} eval acc:{val_avg_acc}')
