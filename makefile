
all : stats

run : main
	./main

stats : get_stats.R main-hip-amdgcn-amd-amdhsa-gfx90a.s
	R --slave -f $<

main main-hip-amdgcn-amd-amdhsa-gfx90a.s &: main.cpp 
	hipcc -save-temps --offload-arch=gfx90a -o main $<

%.ll : %.bc
	/opt/rocm-5.6.1/llvm/bin/llvm-dis -o $@ $<

