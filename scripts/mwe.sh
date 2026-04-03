#!/bin/bash

for AD in Zygote Mooncake Enzyme ForwardDiff; do
    echo "Running with ${AD}..."
     julia --project=.. mwe.jl ${AD} 2> "${AD}_err.log" > "${AD}_out.log"
done