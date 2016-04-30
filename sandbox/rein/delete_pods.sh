#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 150 ]; do
kubectl delete pod "rein-trpo-expl-loco-c1-2016-04-28-15-34-44-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
