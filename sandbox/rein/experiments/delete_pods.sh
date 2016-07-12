#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 51 ]; do
kubectl delete pod "rein-trpo-vime-basic-a-2016-07-09-18-33-54-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
