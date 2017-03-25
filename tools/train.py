import sys
from optparse import OptionParser

sys.path.append('./')

import yolo
import os
from yolo.utils.process_config import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",  
                  help="configure filename")
parser.add_option("-g", "--gpu", dest="gpu",  
                  help="gpu_to_use")
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
else:
  print('please sspecify --conf configure filename')
  exit(0)

if options.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
else:
  print('please specify --gpu 0')
  exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
dataset = eval(dataset_params['name'])(common_params, dataset_params)
net = eval(net_params['name'])(common_params, net_params)
solver = eval(solver_params['name'])(dataset, net, common_params, solver_params)
solver.solve()