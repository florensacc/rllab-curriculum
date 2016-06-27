#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 20 ]; do
kubectl delete pod "rein-trpo-vime-reacher-b-2016-06-23-11-13-31-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
