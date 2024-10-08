#!/usr/bin/env bash

awk 'BEGIN {record=0}
    {
        if ($0~/Analyze> initial cluster index/)
        {
            record=1
        }
        else if ($0~/Analyze> k-means iteration =          1/)
        {
            record=0
        }
        else if (record == 1)
        {
            print $0
        }
    }' output.log
