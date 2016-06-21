#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 20 ]; do
kubectl delete pod "rein-x-brad-trpo-expl-loco-a1-2016-05-19-14-26-02-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
