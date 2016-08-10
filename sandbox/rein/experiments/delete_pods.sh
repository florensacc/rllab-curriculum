#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 61 ]; do
kubectl delete pod "rein-trpo-vime-atari-ent-b-2016-08-08-00-54-39-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
