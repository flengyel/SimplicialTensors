Examples for Computations with simplicial tensor modules

Use Python 3.10 or later. The vector, matrix homology, and horn examples
use the SimplicialTensors revision da5626bfeefd2796c44a8f00e04717d873d52335.
Install it in a virtual environment and run:

    python -m pip install git+https://github.com/flengyel/SimplicialTensors.git@da5626bfeefd2796c44a8f00e04717d873d52335
    python dstm_examples.py

The Petersen example uses sparse rank-shifted tensors from the
SimplicialTensors revision 1995740d13e66d33a74285269a7342ab85bed2d6.
In a separate virtual environment run:

    python -m pip install git+https://github.com/flengyel/SimplicialTensors.git@1995740d13e66d33a74285269a7342ab85bed2d6
    python rank_shifted_homology.py

The script reports normalized chain ranks 10 and 15, incidence rank 9,
and Betti numbers 1 and 6.
