from openvino.inference_engine import IECore
from modules.pre_process import pre_process_openvino
import numpy as np

def prepare(path, device = 'GPU'):
    model_xml = r'{}\model.xml'.format(path)
    model_bin = r'{}\model.bin'.format(path)
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="{}".format(device))  # MULTI:GPU,CPU
    del net
    input_layer = next(iter(exec_net.input_info))
    output_layer = next(iter(exec_net.outputs))
    return exec_net, input_layer, output_layer

def mask_process(img, exec_net, input_layer, output_layer):
    res_list = []
    for x in img:
        x = pre_process_openvino(x)
        res = exec_net.infer(inputs={input_layer: x})
        res = res[output_layer][0].tolist()
        res_list.append(np.argmax(res))
    return res_list

