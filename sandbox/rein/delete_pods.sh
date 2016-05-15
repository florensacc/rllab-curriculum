#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 11 ]; do
kubectl delete pod "rein-x-trpo-expl-basic-d2-par-2016-05-14-18-11-21-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
