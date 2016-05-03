#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 200 ]; do
kubectl delete pod "rein-trpo-expl-loco-c1-2016-04-28-16-23-08-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
