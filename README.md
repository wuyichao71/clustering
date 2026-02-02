# clustering

The program to perform clustering for reweighted gREST/REUS result.

## `clustering.py`

python language
performing clustering.
can add weight

## `plot_comdist_dist_unweight.py`

python language
use `../16_mbar_reus/input/sort_comdist/para{repi}.comdis` to plot distribution.
output is `picture/comdist_unweight.png`

## `plot_comdist_dist_unwtkmeans_unwt.py`

python language
use `../16_mbar_reus/input/sort_comdist/para{repi}.comdis` to plot distribution.
plot distribution without weight
do kmeans without weight
output is `picture/comdist_unwtkmeans_unweight.png`

## `plot_comdist_dist_unwtkmeans_wt.py`

python language
use `../16_mbar_reus/input/sort_comdist/para{repi}.comdis` to plot distribution.
plot distribution with weight
weight is `../16_mbar_reus/result/fes_36/output{repi}.weight`
do kmeans without weight
output is `picture/comdist_unwtkmeans_weight.png`

## `plot_comdist_dist_weight.py`

python language
use `../16_mbar_reus/input/sort_comdist/para{repi}.comdis` to plot distribution.
plot distribution with weight
weight is `../16_mbar_reus/result/fes_36/output{repi}.weight`
output is `picture/comdist_unweight.png`

## `plot_comdist_dist_wtkmeans_wt.py`

python language
use `../16_mbar_reus/input/sort_comdist/para{repi}.comdis` to plot distribution.
plot distribution with weight
weight is `../16_mbar_reus/result/fes_36/output{repi}.weight`
do kmeans with weight
idx file is `output_python.idx`
output is `picture/comdist_wtkmeans_weight.png`

## `test-1/cluster.py`

soft link of `cluster.py`

## `test-1/cluster_unweight.py`

python language
performing kmeans clustering without weight.
input is `../../16_mbar_reus/input/sort_comdist/para{repi}.comdis`
gap is 100
output is `output_unweight_python.idx`

## `test-1/cluster_weight.py`

python language
performing kmeans clustering without weight.
input is `../../16_mbar_reus/input/sort_comdist/para{repi}.comdis`
weight is `../16_mbar_reus/result/fes_36/output{repi}.weight`
gap is 100
output is `output_unweight_python.idx`

## `test-1/comdist.inp`

input of `kmeans_clustering`

## `extract_input.sh`

bash language
submission script
extract initial idx from `output.log`

## `gen_trajectory.py`

python language
use data to generate trajectory.
input is `../16_mbar_reus/input/sort_comdist/para{repi}.comdis`
