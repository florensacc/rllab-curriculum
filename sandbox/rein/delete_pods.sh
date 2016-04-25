#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 111 ]; do
kubectl delete pod "rein-trpo-expl-basic-v3x-2016-04-25-14-01-13-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
