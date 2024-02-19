TVM_ROOT=$(shell cd ../tvm; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core


PKG_CFLAGS = -std=c++17 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I /usr/local/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

.PHONY: clean all

all: lib/resnet_cifar10 lib/resnet_cifar100 lib/VGG16_cifar100  lib/VGG16_cifar10

lib/resnet_cifar10: resnet_cifar10.cpp
	@mkdir -p $(@D)
	$(CXX)  $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)

lib/resnet_cifar100: resnet_cifar100.cpp
	@mkdir -p $(@D)
	$(CXX)  $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)

 lib/VGG16_cifar10: VGG16_cifar10.cpp
	@mkdir -p $(@D)
	$(CXX)  $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)


 lib/VGG16_cifar100: VGG16_cifar100.cpp
	@mkdir -p $(@D)
	$(CXX)  $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)

clean:
	rm -rf lib
