import warnings
from tvm.driver import tvmc
import onnx
from PIL import Image
import numpy as np
import time


warnings.filterwarnings('ignore')

model_path = "/data/ONNX/resnet_cifar10_single.onnx"
onnx_model = onnx.load(model_path)
model = tvmc.load(model_path, shape_dict={"input": [1, 3, 224, 224]}) # Step 1: Load
desired_model_path = 'resnet_cifar10_tvm.onnx'
model.save(desired_model_path)

#model_path = './resnet_cifar10_tvm.onnx'
model_before_tune = tvmc.load(model_path, shape_dict={"input": [1, 3, 224, 224]}) # Step 1: Load
model = model_before_tune
print("start tuning")
start = time.time()
tvmc.tune(model, target="llvm",tuning_records="resnet_cifar10_record.log")#save the tunning result
end = time.time()
print("Time:",(end-start),"s")
package = tvmc.compile(model_before_tune, target="llvm",package_path="resnet_cifar10.tar")#compile
new_package = tvmc.TVMCPackage(package_path="resnet_cifar10.tar")

img_path = "/dataset/cifar-10-batches-py/test/9_9971.png"

resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")
img_data = np.transpose(img_data, (2, 0, 1))

# 根据 ResNet 输入规范进行归一化
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev
img_data = np.expand_dims(norm_img_data, axis=0)
np.savez("image", data=img_data)


start = time.time()
results = tvmc.run(tvmc_package=package, inputs={"input":img_data}, device="cpu")#Run
end = time.time()

print("before tuning:",results.outputs)
print("Time:",(end-start)*1000,"ms")

start = time.time()
results = tvmc.run(tvmc_package=package, inputs={"input":img_data}, device="cpu")#Run
end = time.time()

package = tvmc.compile(model, target="llvm",package_path="resnet_cifar10_tuned.tar",tuning_records="resnet_cifar10_records.log")#compile
new_package = tvmc.TVMCPackage(package_path="resnet_cifar10_tuned.tar")

start = time.time()
results = tvmc.run(tvmc_package=package, inputs={"input":img_data}, device="cpu")#Run
end = time.time()

print("after tuning:",results.outputs)
print("Time:",(end-start)*1000,"ms")
