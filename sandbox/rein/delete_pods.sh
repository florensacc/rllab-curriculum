#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 111 ]; do
kubectl delete pod "rein-trpo-expl-loco-v1x-2016-04-21-18-15-34-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
