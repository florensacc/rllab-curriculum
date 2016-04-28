#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 30 ]; do
kubectl delete pod "rein-trpo-loco-v3y-2016-04-27-14-13-13-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
