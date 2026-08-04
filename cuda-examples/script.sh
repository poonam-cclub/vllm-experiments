#!/bin/bash
for ((i=0; i<40; i++)); do
	./add_small &
done
wait
