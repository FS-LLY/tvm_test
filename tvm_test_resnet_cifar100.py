import onnx
#from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
from tvm.contrib import graph_runtime
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
import time

model_path = "/data/ONNX/resnet_cifar100_single.onnx"
onnx_model = onnx.load(model_path)
np.random.seed(0)
img_path = "/dataset/cifar-100-python/testdir/9_9967.png"

resized_image = Image.open(img_path).resize((224, 224),resample=Image.Resampling.BILINEAR)
#resized_image.save("./img1.png")
img_data = np.asarray(resized_image).astype("float32")
img_data = np.transpose(img_data, (2, 0, 1))
img_data = np.array(img_data)[:,:,::-1]

# 根据 ResNet 输入规范进行归一化
imagenet_mean = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev
img_data = np.expand_dims(norm_img_data, axis=0)

x = img_data
local_demo = False
if local_demo:
    target = tvm.target.Target("llvm")
    dylib_path = "../tvm_output_lib/resnet_cifar100.so"
    dylib_path_tuned = "../tvm_output_lib/resnet_cifar100_tuned.so"
else:
    target = tvm.target.arm_cpu("rasp5b")
    dylib_path = "../tvm_output_lib/resnet_cifar100_aarch.tar"
    dylib_path_tuned = "../tvm_output_lib/resnet_cifar100_aarch_tuned.tar"
input_name = "input"
shape_dict = {input_name: img_data.shape}


mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
with tvm.transform.PassContext(opt_level=1):
    executor = relay.build_module.create_executor(
        "graph", mod, tvm.cpu(0), target, params
    ).evaluate()
lib = relay.build(mod, target=target, params=params)
lib.export_library(dylib_path)
dev = tvm.device(str(target), 0)
module  = graph_executor.GraphModule(lib["default"](dev))


#old version codes

'''
with relay.build_config(opt_level=2):
    graph, lib, params = relay.build_module.build(mod, target, params=params)

print("Output model files")
libpath = "../tvm_output_lib/resnet_cifar10.so"
lib.export_library(libpath)

graph_json_path = "../tvm_output_lib/resnet_cifar10.json"
with open(graph_json_path, 'w') as fo:
    fo.write(graph)

param_path = "../tvm_output_lib/resnet_cifar10.params"
with open(param_path, 'wb') as fo:
    fo.write(relay.save_param_dict(params))

loaded_json = open(graph_json_path).read()
loaded_lib = tvm.runtime.load_module(libpath)
loaded_params = bytearray(open(param_path, "rb").read())

ctx = tvm.cpu()

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.set_input("input", x)
module.run()
out_deploy = module.get_output(0).asnumpy()

print(out_deploy)
'''
dtype = "float32"
if local_demo:
    module.set_input("input",tvm.nd.array(x.astype(dtype)))
    start = time.time()
    module.run()
    end = time.time()
    output_shape = (1,100)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
    print("before tuning:")
    print(tvm_output)
    print("Time:",(end-start)*1000,"ms")

print("start tunning")
number = 10
repeat = 1
min_repeat_ms = 0  # 调优 CPU 时设置为 0
timeout = 3600  # 秒

# 创建 TVM 运行器
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)
tuning_option = {
    "tuner": "xgb",
    "trials": 1500,
    "early_stopping": 200,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "resnet18_cifar100_tunning.json",
}
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    
    # choose tuner
    tuner = "xgb"
    tuner_obj = XGBTuner(task, loss_type="reg")

    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )

print("tunning finish.")
with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
#lib = relay.build(mod, target=target, params=params)

lib.export_library(dylib_path_tuned)

if local_demo:
    module.set_input("input",tvm.nd.array(x.astype(dtype)))
    start = time.time()
    module.run()
    end = time.time()
    output_shape = (1,100)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
    print("after tuning:")
    print(tvm_output)
    print("Time:",(end-start)*1000,"ms")