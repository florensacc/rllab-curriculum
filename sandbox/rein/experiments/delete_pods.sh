#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 61 ]; do
kubectl delete pod "rein-trpo-atari-b-2016-08-03-14-23-50-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
