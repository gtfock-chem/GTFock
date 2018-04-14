Set `WORK_TOP` to where you will install the files.



## Compiling Required Libraries

Please compile required libraries in the following sequence.

### 1.(a) OptERD (deprecated)

Notice: the compatibility of OptERD and GTFock is not tested after GTFock commit version e8398cb ([GTFock commit versions](https://github.com/gtfock-chem/gtfock/commits/master)), you can skip this if you do not need to test OptERD's performance

```shell
cd $WORK_TOP
git clone https://github.com/gtfock-chem/OptErd_Makefile.git
cd OptErd_Makefile
# Adjust the make.in according to your system
```

### 1.(b) Simint

Notice: If possible, use ICC 17 instead of ICC 18 to compile Simint. It seems that there are some problems with Simint compiled by ICC 18.

```shell
# Build Simint source code generator
cd $WORK_TOP
git clone https://github.com/gtfock-chem/simint-generator.git
cd simint-generator
mkdir build
cd build
CC=icc CXX=icpc cmake ../
make -j16
cd ..

# Generate Simint source code
# If your system does not use python 3 as default python interpretor, you can also use python2 to run the generating script
# Run ./create.py --help to see the details of the parameters
./create.py -g build/generator/ostei -l 3 -p 3 -d 0 -ve 4 -vg 5 -he 4 -hg 5 gtfock-simint
mv gtfock-simint ../

# Compile Simint
cd ../gtfock-simint  # Should at $WORK_TOP/gtfock-simint
# For KNL, use other directory name and SIMINT_VECTOR variable for other architecture; see the README file in Simint directory
# Don't set SIMINT_C_FLAGS if you do not need to profile or debug
mkdir build-avx512   
CC=icc CXX=icpc cmake ../ -DSIMINT_VECTOR=micavx512 -DSIMINT_C_FLAGS="-O3;-g" -DCMAKE_INSTALL_PREFIX=./install
make -j16 install
```



### 2. libcint

```shell
cd $WORK_TOP
git clone https://github.com/gtfock-chem/libcint.git
cd libcint
# Adjust the Makefile according to the directory you compiled Simint (and OptERD) and your system
make libcint.a 
```



### 3. ARMCI-MPI

The following commands will use ICC and Intel MPI to compile ARMCI-MPI. You can use other compilers and MPI environments. 

```shell
cd $WORK_TOP
git clone git://git.mpich.org/armci-mpi.git 
cd armci-mpi
git checkout mpi3rma
./autogen.sh
mkdir build
cd build
../configure CC=mpiicc --prefix=$WORK_TOP/ARMCI-MPI
make -j16 install
```

For Cori, use `CC=cc` to replace `CC=mpicc`. 



### 4. Global Array

The following commands will use ICC and Intel MPI to compile ARMCI-MPI. You can use other compilers and MPI environments. Make sure that the compiler and MPI environment are the same as compiling ARMCI-MPI.

```shell
cd $WORK_TOP
# Dont's use the latest version of Global Array, we haven't test it yet
wget http://hpc.pnl.gov/globalarrays/download/ga-5-3.tgz
tar xzf ga-5-3.tgz
cd ga-5-3
# Carefully set the mpi executables that you want to use
# Recommend using a build dir
mkdir build
cd build
../configure CC=mpiicc MPICC=mpiicc CXX=mpiicpc MPICXX=mpiicpc F77=mpiifort MPIF77=mpiifort --with-mpi --with-armci=$WORK_TOP/ARMCI-MPI --prefix=$WORK_TOP/GAlib
make -j16 install
```

For Cori, use
```shell
CC=cc MPICC=cc CXX=CC MPICXX=CC F77=ftn MPIF77=ftn
```
to replace 
```shell
CC=mpiicc MPICC=mpiicc CXX=mpiicpc MPICXX=mpiicpc F77=mpiifort MPIF77=mpiifort
```
For `configure`. 




## Compiling GTFock

Clone GTFock from GitHub:

```shell
cd $WORK_TOP
git clone https://github.com/gtfock-chem/gtfock.git
cd gtfock
```

### Compiling GTFock on Regular Cluster

If your computing platform does not contain special network hardward or settings, you should compile ARMCI-MPI and Global Array yourself and then compile GTFock. Modify `make.in` according to the configuration of your system and the path of required libraries. File `make.in.KNL5` is a sample configuration file that used to compile GTFock on a single Xeon Phi Kinghts Landing node. Make sure that the compiler and MPI environment are the same as compiling ARMCI-MPI and Global Array.

### Compiling GTFock on Cori

You can use ICC + Intel MPI to compile ARMCI-MPI, Global Array and GTFock on Cori, but it is likely that you cannot run the program on multiple nodes. 

To run GTFock on Cori using multiple nodes, you should use Cray compiler wrapper, Cray MPI and Cray SciLib (providing ScaLAPACK and BLASC). Use Cray MPI to compile your own ARMCI-MPI and Global Arrays, DO NOT use the Global Arrays provided by Cray (the `cray-ga` module). Cray compiler wrapper, Cray SciLib and Cray MPI are loaded by default. Use `make.in.Cori` as a template to modify the `make.in`, then you can compile the GTFock.


The gtfock libraries will be installed in `$WORK_TOP/gtfock/install/`, The example SCF code can be found in `$WORK_TOP/gtfock/pscf/`.



## Running Example SCF program

Use the following command to run the example SCF program:

```shell
mpirun -np <nprocs> $WORK_TOP/gtfock/pscf/scf <basis> <xyz> \
<nprow> <npcol> <np2> <ntasks> <niters>
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
* `nprow` x `npcol` must be equal to `nprocs`
* `np2` x `np2` x `np2` should be close to `nprocs` but must be smaller than nprocs
* suggested values for `ntasks`: 3, 4, 5

To run the example SCF program on Cori,  `Cori-KNL-9N.pbs`, `Cori-KNL-3N.pbs` and `Cori-KNL-1N.pbs` are three example slurm batch file. For more information about writing job script for Cori, please refer to [this page](http://www.nersc.gov/users/computational-systems/cori/running-jobs/).