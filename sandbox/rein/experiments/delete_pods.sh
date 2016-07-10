#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 21 ]; do
kubectl delete pod "rein-trpo-vime-venture-a-2016-07-09-11-17-59-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
