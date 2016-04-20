#!/bin/bash 
COUNTER=0
while [  $COUNTER -lt 401 ]; do
kubectl delete pod "rein-trpo-expl-basic-v6-2016-04-19-14-44-16-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
