Set `WORK_TOP` to where you will install the files.



## Compiling Required Libraries

Please compile required libraries in the following sequence.

### 1. Simint

Notice: If possible, use ICC 17 instead of ICC 18 to compile Simint. It seems that there are some problems with Simint compiled by ICC 18. ICC 19 works well. 

```shell
# Build Simint source code generator
cd $WORK_TOP
git clone https://github.com/simint-chem/simint-generator.git
cd simint-generator
mkdir build && cd build
CC=icc CXX=icpc cmake ../
make -j16
cd ..

# Generate Simint source code (requires Python3)
# Run ./create.py --help to see the details of the parameters
./create.py -g build/generator/ostei -l 5 -p 4 -d 0 -ve 4 -vg 5 -he 4 -hg 5 simint
mv simint ../

# Compile Simint
cd ../simint  # Should at $WORK_TOP/simint
# See the README file in Simint directory to see which SIMINT_VECTOR variable you should use
# Commonly used SIMINT_VECTOR: commonavx512, avx2
mkdir build-avx512
CC=icc CXX=icpc cmake ../ -DSIMINT_VECTOR=commonavx512 -DCMAKE_INSTALL_PREFIX=./install
make -j16 install
```



### 2. libcint

```shell
cd $WORK_TOP
git clone https://github.com/gtfock-chem/libcint.git
cd libcint
# Adjust the Makefile according to the directory you compiled Simint and your system
# No need to modify any ERD_* variables since we do not use OptERD now
make libcint.a 
```



### 3. GTMatrix

The following commands will use ICC and Intel MPI to compile GTMatrix. You can use other compilers and MPI environments. 

```shell
cd $WORK_TOP
git clone https://github.com/gtfock-chem/GTMatrix.git
cd GTMatrix
make
```

On Cori, replace `MPICC=mpiicc` with `MPICC=cc` in `Makefile`.



## Compiling GTFock

Clone GTFock from GitHub:

```shell
cd $WORK_TOP
git clone https://github.com/gtfock-chem/gtfock.git
cd gtfock
```

### Compiling GTFock 

Modify `make.in` according to the configuration of your system and the path of required libraries. Make sure that the compiler and MPI environment are the same as compiling GTMatrix.

### Compiling GTFock on Cori

You can use ICC + Intel MPI to compile GTFock and GTFock on Cori, but it is likely that you cannot run the program on multiple nodes. 

To run GTFock on Cori using multiple nodes, you should use Cray compiler wrapper, Cray MPI and Cray SciLib (providing ScaLAPACK and BLASC). Cray compiler wrapper, Cray SciLib and Cray MPI are loaded by default. Modify the `make.in` before compile GTFock on Cori.


The gtfock libraries will be installed in `$WORK_TOP/gtfock/install/`, The example SCF code can be found in `$WORK_TOP/gtfock/pscf/`.



## Running Example SCF program

Use the following command to run the example SCF program:

```shell
mpirun -np <nprocs> $WORK_TOP/gtfock/pscf/scf <basis> <xyz> \
<nprow> <npcol> <np2> <ntasks> <niters>
# Example:
# mpirun -np 8 pscf/scf data/cc-pvdz/cc-pvdz.gbs data/alkane/alkane_62.xyz 4 2 2 5 10 
```

Parameters:
* `nprocs`: Number of MPI processes
* `basis`:  Basis set file
* `xyz`: xyz file for chemcial system
* `nprow`:  Number of MPI processes per row
* `npcol`:  Number of MPI processes per col
* `np2`: Number of MPI processes per dimension (of cube) for purification
* `ntasks`: Each MPI process has `ntasks` x `ntasks` tasks
* `niters`: Max number of SCF iterations

Note:
* `nprow` x `npcol` must equals `nprocs`
* `np2` x `np2` x `np2` should be close to `nprocs` but must not be larger than nprocs
* suggested values for `ntasks`: 3, 4, 5
