#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 61 ]; do
kubectl delete pod "rein-trpo-vime-atari-pxl-g-2016-07-28-17-29-15-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
