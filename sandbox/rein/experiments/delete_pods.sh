#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 61 ]; do
kubectl delete pod "rein-trpo-vime-atari-c-2016-08-05-12-43-39-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
