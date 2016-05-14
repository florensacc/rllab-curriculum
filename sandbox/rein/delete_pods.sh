#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 11 ]; do
kubectl delete pod "rein-x-trpo-expl-basic-b1-2016-05-13-20-48-30-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
