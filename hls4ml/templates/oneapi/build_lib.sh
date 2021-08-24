#!/bin/bash
CC=dpcpp
source /opt/intel/inteloneapi/setvars.sh

CFLAGS="-O3 -fpic -std=c++11"
LDFLAGS="-L${DNNLROOT}/lib"
INCFLAGS="-I${DNNLROOT}/include"
GLOB_ENVS="-DDNNL_CPU_RUNTIME=SYCL -DDNNL_GPU_RUNTIME=SYCL"
PROJECT=myproject

${CC} ${CFLAGS} ${INCFLAGS} -c firmware/${PROJECT}.cpp -o ${PROJECT}.o -DDNNL_USE_DPCPP_USM
${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o -o firmware/lib${PROJECT}.so
rm -f *.o