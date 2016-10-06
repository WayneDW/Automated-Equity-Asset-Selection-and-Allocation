#!/bin/bash
fileName="mid_cap_5_factor_const_VaR_2010-02_to_2015-07_0_1_bounded"
a=1
while [ "$a" -le 30 ]
do
    echo $a "times"
    if [ $a -le 9 ]
    then
        ./main.py >> ./output/5_factor/${fileName}_0${a}
        tail -3 ./output/5_factor/${fileName}_0${a}
    else 
        ./main.py >> ./output/5_factor/${fileName}_${a}
        tail -3 ./output/5_factor/${fileName}_${a}
    fi
    a=`expr $a + 1`
done

