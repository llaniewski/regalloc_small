#include <stdio.h>
#include <hip/hip_runtime.h>


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
hipError_t HandleError( hipError_t err, const char *file, int line ) {
    if (err != hipSuccess) {
        printf("%s in %s at line %d\n", hipGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
	return err;
}


typedef double real_t;

struct Container {
    real_t * tab;
    int nx,ny,nz,nd;
};

__constant__ Container constContainer;
__constant__ real_t Relax;

struct Access {
    const int x, y, z;
    const int nx, ny, nz;
    __device__ Access(int x_, int y_, int z_) : 
        x(x_), y(y_), z(z_), 
        nx(constContainer.nx), ny(constContainer.ny), nz(constContainer.nz) {

    };
    __device__ real_t getF(int dx, int dy, int dz, int d) const {
        int x_ = x + dx;
        int y_ = y + dy;
        int z_ = z + dz;
        if (x_ <  0)  x_ = x_ + nx;
        if (x_ >= nx) x_ = x_ - nx;
        if (y_ <  0)  y_ = y_ + ny;
        if (y_ >= ny) y_ = y_ - ny;
        if (z_ <  0)  z_ = z_ + nz;
        if (z_ >= nz) z_ = z_ - nz;
        return constContainer.tab[x_ + nx*(y_ + ny*(z_+nz*d))];
    }
    __device__ void push(const real_t& val, int d) const {
        constContainer.tab[x + nx*(y + ny*(z+nz*d))] = val;
    }
    
};

const int d3q27_ex[27] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
const int d3q27_ey[27] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1};
const int d3q27_ez[27] = {0, 0, 0, 0, 0, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1};

template <class A> struct Node {
    const A& acc;
    real_t f[27];
    __device__ Node(const A& acc_) : acc(acc_) {};
    __device__ void RunElement() {
        #pragma unroll
        for (int i=0; i<27; i++) {
            f[i] = acc.getF(d3q27_ex[i],d3q27_ey[i],d3q27_ez[i],i);
        }
        real_t dens = 0;
        #pragma unroll
        for (int i=0; i<27; i++) {
            dens = dens + f[i];
        }
        #pragma unroll
        for (int i=0; i<27; i++) {
            f[i] = Relax * f[i] + (1.0 - Relax) * dens/27.0;
            
        }
        #pragma unroll
        for (int i=0; i<27; i++) {
            acc.push(f[i], i);
        }        
    }
    __device__ void RunElementNoUnroll() {
        #pragma nounroll
        for (int i=0; i<27; i++) {
            f[i] = acc.getF(d3q27_ex[i],d3q27_ey[i],d3q27_ez[i],i);
        }
        real_t dens = 0;
        #pragma nounroll
        for (int i=0; i<27; i++) {
            dens = dens + f[i];
        }
        #pragma nounroll
        for (int i=0; i<27; i++) {
            f[i] = Relax * f[i] + (1.0 - Relax) * dens/27.0;
        }
        #pragma nounroll
        for (int i=0; i<27; i++) {
            acc.push(f[i], i);
        }        
    }
};

__global__ void Kernel() {
    int x_ = threadIdx.x + blockIdx.x*blockDim.x;
	int y_ = threadIdx.y + blockIdx.y*blockDim.y;
	int z_ =               blockIdx.z;
    Access acc(x_,y_,z_);
    {
        Node< Access > now(acc);
        now.RunElement();
    }
}

__global__ void KernelDouble() {
    int x_ = threadIdx.x + blockIdx.x*blockDim.x;
	int y_ = threadIdx.y + blockIdx.y*blockDim.y;
	int z_ =               blockIdx.z;
    Access acc(x_,y_,z_);
    {
        Node< Access > now(acc);
        now.RunElement();
    }
    {
        Node< Access > now(acc);
        now.RunElement();
    }
}


__global__ void KernelQuad() {
    int x_ = threadIdx.x + blockIdx.x*blockDim.x;
	int y_ = threadIdx.y + blockIdx.y*blockDim.y;
	int z_ =               blockIdx.z;
    Access acc(x_,y_,z_);
    {
        Node< Access > now(acc);
        now.RunElement();
    }
    {
        Node< Access > now(acc);
        now.RunElement();
    }
    {
        Node< Access > now(acc);
        now.RunElement();
    }
    {
        Node< Access > now(acc);
        now.RunElement();
    }
}


__global__ void KernelNoUnroll() {
    int x_ = threadIdx.x + blockIdx.x*blockDim.x;
	int y_ = threadIdx.y + blockIdx.y*blockDim.y;
	int z_ =               blockIdx.z;
    Access acc(x_,y_,z_);
    {
        Node< Access > now(acc);
        now.RunElementNoUnroll();
    }
}


int main () {
    Container container;
    container.nx = 128;
    container.ny = 128;
    container.nz = 128;
    container.nd = 27;
    HANDLE_ERROR( hipMalloc(&container.tab, container.nx*container.ny*container.nz*container.nd*sizeof(real_t)) );
    HANDLE_ERROR( hipMemcpyToSymbol(HIP_SYMBOL(constContainer),&container,sizeof(Container),0,hipMemcpyHostToDevice) );
    real_t relax_value = 0.5;
    HANDLE_ERROR( hipMemcpyToSymbol(HIP_SYMBOL(Relax),&relax_value,sizeof(real_t),0,hipMemcpyHostToDevice) );
    Kernel <<< dim3(container.nx/64,container.ny/2,container.nz), dim3(64,2) >>>();
    HANDLE_ERROR( hipFree(container.tab) );
    printf("Finished\n");
    return 0;
}