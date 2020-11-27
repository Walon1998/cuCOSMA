#!/bin/bash

echo $CUDA_VISIBLE_DEVICES


print_help(){
    echo "Usage: "
    echo "-d , --debug                          Run the program in debug mode (default)"
    echo "-r , --release                        Run the program in release mode"
    echo "-b , --bench                          Run benchmarks"
    echo "-c , --correct                        Run correctness tests"
    echo "-f <filename>, --file <filename>      Specify file to be used (default: benchmark.csv)"
    echo "-cfg <arch>                           Create a cfg for the specified architecture"
    echo "-env , --environment                  Collect environment information"
    echo "-mem , --memcheck                     Run with cuda-memcheck"
    echo "-p , --plot                           Create plots of output"
    
    echo "-h, --help:                           Display help"
    echo ""
    exit 0

}

print_wrong_config(){

    echo "#define TYPE $TYPE" 
    echo "#define VECTORTYPE2 $VECTORTYPE2" 
    echo "#define VECTORTYPE4 $VECTORTYPE4" 

    #echo "#define TYPE_STRING \"$TYPE\"" >> $OUTPUT


    echo "#define M $M" 
    echo "#define N $N" 
    echo "#define K $K" 
    echo "#define THREADBLOCK_TILE_M $THREADBLOCK_TILE_M" 
    echo "#define THREADBLOCK_TILE_N $THREADBLOCK_TILE_N" 
    echo "#define THREADBLOCK_TILE_K $THREADBLOCK_TILE_K" 
    echo "#define LOAD_K $LOAD_K" 
    echo "#define WARP_TILE_M $WARP_TILE_M" 
    echo "#define WARP_TILE_N $WARP_TILE_N" 
    echo "#define THREAD_TILE_M $THREAD_TILE_M" 
    echo "#define THREAD_TILE_N $THREAD_TILE_N" 


    echo "#define A_OFFSET $A_OFFSET" 
    echo "#define B_OFFSET $B_OFFSET"

    echo "#define SWIZZLE $SWIZZLE"



    echo "#define SPLIT_K $SPLIT_K" 

    echo "#define ADDITIONAL_OCCUPANCY_WARP $ADDITIONAL_OCCUPANCY_WARP" 
    echo "#define ADDITIONAL_OCCUPANCY_SM $ADDITIONAL_OCCUPANCY_SM" 

    echo "#define ALPHA $ALPHA" 

    echo "#define BETA $BETA" 
    echo " "
    exit
}

# Default values
filename=benchmark.csv
env=0
debug=1
test_correct=0
benchmark=0
INPUT=../benchmark.csv
memcheck=0
cfg=0
arch=52
plot=0

export OMP_NUM_THREADS=$(nproc)
export CUDA_AUTO_BOOST=0

module load cuda/11.0
module load gcc/8

if [ $# -eq 0 ]; then
    print_help
fi



while [ "$1" != "" ]; do
    case $1 in
        -r | --release )        debug=0
                                export LSB_OUTPUT_FORMAT=pretty
                                ;;
        -d | --debug )          debug=1
                                ;;
        -b | --bench )          benchmark=1
                                ;;
        -c | --correct )        test_correct=1
                                ;;      
        -env | --environment)   env=1
                                ;;    
        -mem | --memcheck)      memcheck=1
                                ;;    
        -p | --plot)            plot=1
                                ;;                           
        -cfg )                  cfg=1
                                shift
                                arch=$1
                                ;;                                                                                                
        -h | --help )           print_help                                
                                ;;

        -f | --file )           shift
                                INPUT="../"$1
                                filename="$(basename $1)"
                                ;;                       
        * )                     print_help
    esac
    shift
done



if [ $env -eq 1 ] ; then
    dirname=${filename}_$(hostname)_$(date +"%F_%X")
    mkdir $dirname
    cd $dirname
    nvcc --version > nvcc.txt
    gcc --version > gcc.txt
    lscpu > cpu.txt
    lsmem > mem.txt
    hostnamectl > host.txt
    nvidia-smi -q > nvidia-smi.txt



    curr=$(pwd)
    cd ../src/Util/deviceQuery
    make clean > /dev/null
    make > /dev/null
    ./deviceQuery > ${curr}/gpu.txt

    cd $curr
    cd ..
fi

if [ $cfg -eq 1 ]; then
    cd ./Release
    make clean
    make
    cuobjdump ./cuCOSMA -xelf $arch
    nvdisasm -cfg ./cuCOSMA.7.sm_$arch.cubin | dot -o$arch.svg  -Tsvg
    mv $arch.svg ..
    exit 1
fi


if [ $debug -eq 1 ]; then
    cd ./Debug
else
    cd ./Release
fi

rm config_* &> /dev/null
rm nvprof_* &> /dev/null





if [ $test_correct -eq 0 ] && [ $benchmark -eq 0 ] && [ $cfg -eq 0 ] ; then
    print_help
fi





compiler_out="../console.log"
if [ -f "$compiler_out" ]; then
    rm $compiler_out
fi

echo "Working on: " $INPUT
OUTPUT=../src/config.h
IFS=','
counter=0
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read ADDITIONAL_OCCUPANCY_SM	TYPE	VECTORTYPE2	VECTORTYPE4	M	N	K	THREADBLOCK_TILE_M	THREADBLOCK_TILE_N	THREADBLOCK_TILE_K	LOAD_K	WARP_TILE_M	WARP_TILE_N	THREAD_TILE_M	THREAD_TILE_N	SPLIT_K	ALPHA	BETA  SWIZZLE   A_OFFSET   B_OFFSET  ATOMIC_REDUCTION





do 
    if [ "$M" =  "M" ] || [ "$M" =  "" ]; then
        continue
    fi

  	echo "/*" > $OUTPUT
    echo " * config.h " >> $OUTPUT
    echo " *" >> $OUTPUT
    echo " *  Created on: $(date)" >> $OUTPUT
    echo " *      Author: Automatically generated" >> $OUTPUT
    echo " */" >> $OUTPUT

    echo "#ifndef CONFIG_H_" >> $OUTPUT
    echo "#define CONFIG_H_" >> $OUTPUT


    echo "#define TYPE $TYPE" >> $OUTPUT
    echo "#define VECTORTYPE2 $VECTORTYPE2" >> $OUTPUT
    echo "#define VECTORTYPE4 $VECTORTYPE4" >> $OUTPUT

    #echo "#define TYPE_STRING \"$TYPE\"" >> $OUTPUT


    echo "#define M $M" >> $OUTPUT
    echo "#define N $N" >> $OUTPUT
    echo "#define K $K" >> $OUTPUT
    echo "#define THREADBLOCK_TILE_M $THREADBLOCK_TILE_M" >> $OUTPUT
    echo "#define THREADBLOCK_TILE_N $THREADBLOCK_TILE_N" >> $OUTPUT
    echo "#define THREADBLOCK_TILE_K $THREADBLOCK_TILE_K" >> $OUTPUT
    echo "#define LOAD_K $LOAD_K" >> $OUTPUT
    echo "#define WARP_TILE_M $WARP_TILE_M" >> $OUTPUT
    echo "#define WARP_TILE_N $WARP_TILE_N" >> $OUTPUT
    echo "#define THREAD_TILE_M $THREAD_TILE_M" >> $OUTPUT
    echo "#define THREAD_TILE_N $THREAD_TILE_N" >> $OUTPUT

    echo "#define A_OFFSET $A_OFFSET" >> $OUTPUT
    echo "#define B_OFFSET $B_OFFSET" >> $OUTPUT

    echo "#define SWIZZLE $SWIZZLE" >> $OUTPUT



    echo "#define SPLIT_K $SPLIT_K" >> $OUTPUT
    echo "#define ATOMIC_REDUCTION $ATOMIC_REDUCTION" >> $OUTPUT

    echo "#define ADDITIONAL_OCCUPANCY_SM $ADDITIONAL_OCCUPANCY_SM" >> $OUTPUT

    echo "#define ALPHA $ALPHA" >> $OUTPUT

    echo "#define BETA $BETA" >> $OUTPUT




    if [ $test_correct -eq  1 ]; then
        echo "#define CORRECTNESS_TEST" >> $OUTPUT        
    fi

    if [ $benchmark -eq  1 ]; then
        echo "#define BENCHMARK" >> $OUTPUT
    fi

    echo "#endif /* CONFIG_H_ */" >> $OUTPUT

    if [ $debug -eq  1 ]; then
        make clean 
        make

        if [ $? -ne 0 ]; then
            print_wrong_config
            continue
        fi    

        if [ $memcheck -eq 1 ]; then
            cuda-memcheck ./cuCOSMA
        else
            ./cuCOSMA
        fi
        
    else 
        make clean &>> $compiler_out
        make &>> $compiler_out

        if [ $? -ne 0 ]; then
            print_wrong_config
            continue
        fi  

        if [ $memcheck -eq 1 ]; then
            cuda-memcheck ./cuCOSMA |& tee --append $compiler_out >&1
        else

        export nvprof_logfile=${counter}"_"$M"x"$N"x"$K".csv"



        nvprof --concurrent-kernels off --log-file "nvprof_%q{nvprof_logfile}" --csv  --profile-from-start off --profile-api-trace all --print-gpu-trace   ./cuCOSMA |& tee --append $compiler_out >&1

        cp $OUTPUT "config_"$nvprof_logfile

        # nvprof --concurrent-kernels off --log-file out.csv  --profile-from-start off --profile-api-trace none --csv --print-gpu-trace  ./cuCOSMA
        # nvprof --concurrent-kernels off  --profile-from-start off --profile-api-trace none   ./cuCOSMA
         #   ./cuCOSMA |& tee --append $compiler_out >&1

        fi

      
        
    fi

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
        print_wrong_config
        continue
        fi  


counter=$(( counter + 1 ))

done < $INPUT





if [ $env -eq 1 ]; then


    if [ $debug -eq 0 ]; then
        mv $compiler_out ../$dirname
    fi


    if [ $benchmark -eq 1 ]; then



        mv nvprof_* ../$dirname
        mv config_* ../$dirname

        lsbfiles=../$dirname



   

    fi


else

    if [ $benchmark -eq 1 ]; then



        mv nvprof_* ../$dirname
        mv config_* ../$dirname

        lsbfiles=../
    fi

fi



if [ $plot -eq 1 ] ; then
    exit -1
    cd ../R
    mkdir temp

    for f in $lsbfiles/lsb.*; do
    cp $f temp

    ./lsbpp.R temp/  violinplot Impl time --xlabel="CutlassCuCOSMA--Cublas--CutlassDefault" --ylabel="Microsecond" --plotout=${f}_violinplot.pdf &> /dev/null
    ./lsbpp.R temp/  boxplot Impl time --xlabel="CutlassCuCOSMA--Cublas--CutlassDefault" --ylabel="Microsecond" --plotout=${f}_boxplot.pdf &> /dev/null
    rm temp/*
    done

    rm -rf temp



fi