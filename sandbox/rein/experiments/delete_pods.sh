#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 61 ]; do
kubectl delete pod "rein-trpo-vime-atari-42x52-var-a-2016-08-26-14-09-59-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
