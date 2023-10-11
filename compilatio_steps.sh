/opt/rocm-5.6.1/llvm/bin/clang-16 \
	-cc1 \
	-triple amdgcn-amd-amdhsa \
	-aux-triple x86_64-unknown-linux-gnu \
	-E \
	-save-temps=cwd \
	-disable-free \
	-clear-ast-before-backend \
	-disable-llvm-verifier \
	-discard-value-names \
	-main-file-name main.cpp \
	-mrelocation-model pic \
	-pic-level 1 \
	-fhalf-no-semantic-interposition \
	-mframe-pointer=none \
	-fno-rounding-math \
	-mconstructor-aliases \
	-aux-target-cpu x86-64 \
	-fcuda-is-device \
	-mllvm \
	-amdgpu-internalize-symbols \
	-fcuda-allow-variadic-functions \
	-fvisibility=hidden \
	-fapply-global-visibility-to-externs \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/hip.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/ocml.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/ockl.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_daz_opt_off.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_unsafe_math_off.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_finite_only_off.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_wavefrontsize64_on.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_isa_version_90a.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_abi_version_500.bc \
	-target-cpu gfx90a \
	-debugger-tuning=gdb \
	-resource-dir /opt/rocm-5.6.1/llvm/lib/clang/16.0.0 \
	-internal-isystem /opt/rocm-5.6.1/llvm/lib/clang/16.0.0/include/cuda_wrappers \
	-idirafter /opt/rocm-5.6.1/include \
	-include __clang_hip_runtime_wrapper.h \
	-isystem /opt/rocm-5.6.1/include \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/x86_64-linux-gnu/c++/12 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/backward \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/x86_64-linux-gnu/c++/12 \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/backward \
	-internal-isystem /opt/rocm-5.6.1/llvm/lib/clang/16.0.0/include \
	-internal-isystem /usr/local/include \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../x86_64-linux-gnu/include \
	-internal-externc-isystem /usr/include/x86_64-linux-gnu \
	-internal-externc-isystem /include \
	-internal-externc-isystem /usr/include \
	-internal-isystem /opt/rocm-5.6.1/llvm/lib/clang/16.0.0/include \
	-internal-isystem /usr/local/include \
	-internal-isystem /usr/lib/gcc/x86_64-linux-gnu/12/../../../../x86_64-linux-gnu/include \
	-internal-externc-isystem /usr/include/x86_64-linux-gnu \
	-internal-externc-isystem /include \
	-internal-externc-isystem /usr/include \
	-O3 \
	-fdeprecated-macro \
	-fno-autolink \
	-fdebug-compilation-dir=/workspaces/regalloc_small \
	-ferror-limit 19 \
	-fhip-new-launch-api \
	-fgnuc-version=4.2.1 \
	-fcxx-exceptions \
	-fexceptions \
	-vectorize-loops \
	-vectorize-slp \
	-mllvm \
	-amdgpu-early-inline-all=true \
	-mllvm \
	-amdgpu-function-calls=false \
	-cuid=7c31d38f66754200 \
	-fcuda-allow-variadic-functions \
	-faddrsig \
	-D__GCC_HAVE_DWARF2_CFI_ASM=1 \
	-o main-hip-amdgcn-amd-amdhsa-gfx90a.hipi \
	-x hip main.cpp
/opt/rocm-5.6.1/llvm/bin/clang-16 \
	-cc1 \
	-triple amdgcn-amd-amdhsa \
	-aux-triple x86_64-unknown-linux-gnu \
	-emit-llvm-bc \
	-emit-llvm-uselists \
	-save-temps=cwd \
	-disable-free \
	-clear-ast-before-backend \
	-disable-llvm-verifier \
	-discard-value-names \
	-main-file-name main.cpp \
	-mrelocation-model pic \
	-pic-level 1 \
	-fhalf-no-semantic-interposition \
	-mframe-pointer=none \
	-fno-rounding-math \
	-mconstructor-aliases \
	-aux-target-cpu x86-64 \
	-fcuda-is-device \
	-mllvm \
	-amdgpu-internalize-symbols \
	-fcuda-allow-variadic-functions \
	-fvisibility=hidden \
	-fapply-global-visibility-to-externs \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/hip.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/ocml.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/ockl.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_daz_opt_off.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_unsafe_math_off.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_finite_only_off.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_wavefrontsize64_on.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_isa_version_90a.bc \
	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_abi_version_500.bc \
	-target-cpu gfx90a \
	-mllvm \
	-treat-scalable-fixed-error-as-warning \
	-debugger-tuning=gdb \
	-resource-dir /opt/rocm-5.6.1/llvm/lib/clang/16.0.0 \
	-O3 \
	-fdeprecated-macro \
	-fno-autolink \
	-fdebug-compilation-dir=/workspaces/regalloc_small \
	-ferror-limit 19 \
	-fhip-new-launch-api \
	-fgnuc-version=4.2.1 \
	-fcxx-exceptions \
	-fexceptions \
	-vectorize-loops \
	-vectorize-slp \
	-mllvm \
	-amdgpu-early-inline-all=true \
	-mllvm \
	-amdgpu-function-calls=false \
	-disable-llvm-passes \
	-cuid=7c31d38f66754200 \
	-fcuda-allow-variadic-functions \
	-faddrsig \
	-D__GCC_HAVE_DWARF2_CFI_ASM=1 \
	-o main-hip-amdgcn-amd-amdhsa-gfx90a.bc \
	-x hip-cpp-output main-hip-amdgcn-amd-amdhsa-gfx90a.hipi

/opt/rocm-5.6.1/llvm/bin/llvm-dis \
	-o main-hip-amdgcn-amd-amdhsa-gfx90a.ll \
	main-hip-amdgcn-amd-amdhsa-gfx90a.bc
/opt/rocm-5.6.1/llvm/bin/opt \
	-amdgpu-early-inline-all=true \
	-amdgpu-function-calls=false \
	--amdgpu-codegenprepare \
	-o main-hip-amdgcn-amd-amdhsa-gfx90a-opt.bc \
	main-hip-amdgcn-amd-amdhsa-gfx90a.bc
/opt/rocm-5.6.1/llvm/bin/llvm-dis \
	-o main-hip-amdgcn-amd-amdhsa-gfx90a-opt.ll \
	main-hip-amdgcn-amd-amdhsa-gfx90a-opt.bc


# /opt/rocm-5.6.1/llvm/bin/clang-16 \
# 	-cc1 \
# 	-triple amdgcn-amd-amdhsa \
# 	-aux-triple x86_64-unknown-linux-gnu \
# 	-S \
# 	-save-temps=cwd \
# 	-disable-free \
# 	-clear-ast-before-backend \
# 	-disable-llvm-verifier \
# 	-discard-value-names \
# 	-main-file-name main.cpp \
# 	-mrelocation-model pic \
# 	-pic-level 1 \
# 	-fhalf-no-semantic-interposition \
# 	-mframe-pointer=none \
# 	-fno-rounding-math \
# 	-mconstructor-aliases \
# 	-aux-target-cpu x86-64 \
# 	-fcuda-is-device \
# 	-mllvm \
# 	-amdgpu-internalize-symbols \
# 	-fcuda-allow-variadic-functions \
# 	-fvisibility=hidden \
# 	-fapply-global-visibility-to-externs \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/hip.bc \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/ocml.bc \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/ockl.bc \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_daz_opt_off.bc \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_unsafe_math_off.bc \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_finite_only_off.bc \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_wavefrontsize64_on.bc \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_isa_version_90a.bc \
# 	-mlink-builtin-bitcode /opt/rocm-5.6.1/amdgcn/bitcode/oclc_abi_version_500.bc \
# 	-target-cpu gfx90a \
# 	-mllvm \
# 	-treat-scalable-fixed-error-as-warning \
# 	-debugger-tuning=gdb \
# 	-resource-dir /opt/rocm-5.6.1/llvm/lib/clang/16.0.0 \
# 	-O3 \
# 	-fno-autolink \
# 	-fdebug-compilation-dir=/workspaces/regalloc_small \
# 	-ferror-limit 19 \
# 	-fhip-new-launch-api \
# 	-fgnuc-version=4.2.1 \
# 	-vectorize-loops \
# 	-vectorize-slp \
# 	-mllvm \
# 	-amdgpu-early-inline-all=true \
# 	-mllvm \
# 	-amdgpu-function-calls=false \
# 	-cuid=7c31d38f66754200 \
# 	-fcuda-allow-variadic-functions \
# 	-faddrsig \
# 	-o main-hip-amdgcn-amd-amdhsa-gfx90a.s \
# 	-x ir main-hip-amdgcn-amd-amdhsa-gfx90a.bc

/opt/rocm-5.6.1/llvm/bin/llc \
	--vgpr-regalloc=basic \
	-amdgpu-early-inline-all=true \
	-amdgpu-function-calls=false \
	-O3 \
	-o main-hip-amdgcn-amd-amdhsa-gfx90a.s \
	-x ir main-hip-amdgcn-amd-amdhsa-gfx90a-opt.bc

R --slave -f get_stats.R
