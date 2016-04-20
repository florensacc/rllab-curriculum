#!/bin/bash 
COUNTER=0
while [  $COUNTER -lt 11 ]; do
kubectl delete pod "rein-trpo-basic-v1x-2016-04-19-21-57-09-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
