from openvino.inference_engine import IECore
from modules.pre_process import pre_process_openvino
import numpy as np

def mask_prepare(path, device = 'GPU'):
    model_xml = r'{}\mobilenet.xml'.format(path)
    model_bin = r'{}\mobilenet.bin'.format(path)
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="{}".format(device))  # MULTI:GPU,CPU
    del net
    input_layer = next(iter(exec_net.input_info))
    output_layer = next(iter(exec_net.outputs))
    return exec_net, input_layer, output_layer

def mask_process(img, landmarks, exec_net, input_layer, output_layer):
    res_list = []
    for x in landmarks:
        try:
            x = pre_process_openvino(img, x)
            res = exec_net.infer(inputs={input_layer: x})
            res = res[output_layer][0].tolist()
            res_list.append(np.argmax(res))
            '''
            max_res = np.max(res)
            if np.argmax(res) == 0 :#and max_res >0.8:
                res_list.append(np.argmax(res))
            else:
                res_list.append(1)
            '''
        except:
            print("Fail preprocess")
            res_list.append(np.argmax(0))
    return res_list

