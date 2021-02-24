#!/usr/bin/env bash

# slurm batch script
#SBATCH -o /home/comp/qiangwang/blackjack/once-for-all/width.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH -w hkbugpusrv04,hkbugpusrv05

# Train on Scene Flow training set
PY="/usr/local/bin/python"
MPIPATH="/home/comp/qiangwang/software/openmpi-4.0.1"
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1  \
	--mca btl_tcp_if_include ib0  \
	--mca btl_openib_want_fork_support 1   \
	-x LD_LIBRARY_PATH    \
	-x NCCL_IB_DISABLE=0  \
	-x NCCL_SOCKET_IFNAME=ib0  \
	-x NCCL_DEBUG=INFO  \
	-x HOROVOD_CACHE_CAPACITY=0"
#hosts="-np 4 -H hkbugpusrv05:4"
hosts="-np 8 -H hkbugpusrv04:4,hkbugpusrv05:4"

# train the large super net
#$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
#	$params \
#	$PY train_ofa_stereo.py --task large

# shrink the kernel,depth,width
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
	$params \
	$PY train_ofa_stereo.py --task expand
