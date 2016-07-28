#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 61 ]; do
kubectl delete pod "rein-trpo-vime-atari-pxl-c-2016-07-27-12-08-41-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
