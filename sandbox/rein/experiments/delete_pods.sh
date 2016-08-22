#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 61 ]; do
kubectl delete pod "rein-trpo-atari-84x84-a-2016-08-20-15-03-04-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
