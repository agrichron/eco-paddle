# import torch
# from torch import nn
from layer_factory import get_basic_layer, parse_expr
# import torch.utils.model_zoo as model_zoo
import yaml
import paddle.fluid as fluid

class ECOfull:
    def __init__(self, input, model_path='ECOfull.yaml', num_classes=101,
                       num_segments=4, pretrained_parts='both'):

        super(ECOfull, self).__init__()

        self.num_segments = num_segments

        self.pretrained_parts = pretrained_parts

        manifest = yaml.load(open(model_path))

        layers = manifest['layers']

        self._channel_dict = dict()

        # in_tmp = input
        setattr(self, 'data', input)
        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            id = l['id']
            if op != 'Concat' and op != 'Eltwise' and op != 'InnerProduct':
                in_channel = getattr(self, in_var[0])
                if out_var[0] == 'res3a_2' or out_var[0] == 'global_pool2D_reshape_consensus':
                        in_channel_tmp = fluid.layers.reshape(in_channel, (-1, self.num_segments)+ in_channel.shape[1:])
                        in_channel = fluid.layers.transpose(in_channel_tmp, [0, 2, 1, 3, 4])

                id, out_name, module, in_name = get_basic_layer(l, in_channel, True, num_segments=num_segments)

                setattr(self, id, module)
            elif op == 'Concat':
                try:
                    in_channel = [getattr(self, var) for var in in_var]
                    module = fluid.layers.concat(in_channel, 1, name=id)
                    setattr(self, id, module)
                except:
                    for x in in_channel:
                        print(x.shape)
                    raise 'eeeeee'
            elif op == 'InnerProduct':
                in_channel = getattr(self, in_var[0])
                in_channel = fluid.layers.reshape(in_channel, (-1, in_channel.shape[1]))
                module = get_basic_layer(l, in_channel)
                setattr(self, id, module)
            else:
                x1 = getattr(self, in_var[0])
                x2 = getattr(self, in_var[1])
                module = fluid.layers.elementwise_add(x1, x2, 1)
                setattr(self, id, module)

    def __call__(self):
        return getattr(self, 'fc_final')[2]

if __name__ == '__main__':
    
    main_program = fluid.default_main_program()
    with fluid.program_guard(main_program):
        data = fluid.data(name='data', shape=[16*4,3,224,224])
        net = ECOfull(data)
        out = net()
    
        # prog = fluid.default_main_program()
        print('---'*50)
        # print(prog.to_string(False, True))
        print(main_program.all_parameters())