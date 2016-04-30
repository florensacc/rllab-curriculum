#!/bin/bash 
COUNTER=1
while [  $COUNTER -lt 50 ]; do
kubectl delete pod "rein-ddpg-basic-a1-2016-04-28-15-47-57-"$(printf %04d $COUNTER)
let COUNTER=COUNTER+1 
done
         
