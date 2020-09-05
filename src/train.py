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

best_acc = 0

def read_best_acc():
    filename = Path('models/best_val_acc.txt')
    if filename.exists():
        with open(filename, 'r') as f:
            line = f.readline().strip().split()[-1]
            line = float(line)
            return line
    else:
        return 0
    

def validate(val_reader, feeder, place, program, fetch_list, epoch=0, writer=None):
    global best_acc

    exe = fluid.Executor(place)

    val_avg_loss = []
    val_acc = []
    for data in val_reader():

        avg_cost_, acc_ = exe.run(program, feed=feeder.feed(data), fetch_list=fetch_list)
        val_avg_loss += [avg_cost_]
        val_acc += [acc_]

    val_avg_loss = np.array(val_avg_loss).mean()
    val_avg_acc = np.array(val_acc).mean()
    if writer:
        writer.add_scalar(tag='val/loss', step=(epoch+1), value = val_avg_loss )
        writer.add_scalar(tag='val/acc', step=(epoch+1), value = val_avg_acc )
    print(f'epoch:{epoch+1} val avg_loss:{val_avg_loss} eval acc:{val_avg_acc}')
    fluid.io.save(fluid.default_main_program(), args.snapshot_pref)

    if val_avg_acc > best_acc:
        best_acc = val_avg_acc
        for item in ['.pdmodel', '.pdparams', '.pdopt']:
            src = args.snapshot_pref + item
            dst = args.snapshot_pref + '_best' + item
            shutil.copy(src, dst)
        os.system(f'echo {val_avg_loss} {val_avg_acc} > {Path(args.snapshot_pref).parent}/best_val_acc.txt')


def main():
    global args, best_acc
    args = parser.parse_args()
    
    writer = LogWriter(args.log)
    # writer = None

    cfg = parse_config('config.txt')
    print_configs(cfg, 'TRAIN')

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
    
    #print(main_program.all_parameters())
    reader = KineticsReader('eco', 'train', cfg).create_reader()
    feeder = fluid.DataFeeder([input, label], place)

     # 验证集
    val_reader = KineticsReader('eco', 'valid', cfg).create_reader()

    # 初始化参数
    exe = fluid.Executor(place=place)
    exe.run(start_program)
    

    train_exe = fluid.Executor(place=place)
    
    if 0:
        # fluid.io.load(train_exe, 'models/', filename='eco_full.pdparams')
        fluid.io.load(main_program, 'models/eco_full_best', train_exe)
    # # pre-trained
    else:
        f = open('program_state_dict.pkl', 'rb')
        state_dict = pickle.load(f)
        f.close()
        fluid.io.set_program_state(main_program, state_dict)


    step = 0
    best_acc = read_best_acc()
    for i in range(args.epochs):
        for index, data in enumerate(reader()):
            avg_cost_, acc_ = train_exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost.name, acc.name])

            if (index + 1) % args.print_freq == 0:
                if not writer is None:
                    writer.add_scalar(tag='train/loss', step=step, value=avg_cost_[0])
                    writer.add_scalar(tag='train/acc', step=step, value=acc_[0])
                print(f'epoch:{i+1} step:{index + 1} avg loss:{avg_cost_[0]} acc:{acc_[0]}')
            step += 1

        if (i + 1) % args.eval_freq == 0:
            fetch_list = [avg_cost.name, acc.name]
            validate(val_reader, feeder, place, eval_program, fetch_list, epoch=i, writer=writer)
        

if __name__=='__main__':
    main()
