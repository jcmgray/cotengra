<p align="left"><img src="https://imgur.com/OM5XyaD.png" alt="cotengra" width="300px"></p>

`cotengra` is a python library for contracting tensor networks or einsum
expressions involving large numbers of tensors - the main docs can be found
at [cotengra.readthedocs.io](https://cotengra.readthedocs.io/).
Some of the key feautures of `cotengra` include:

* an explicit **contraction tree** object that can be flexibly built, modified and visualized
* a **'hyper optimizer'** that samples trees while tuning the generating meta-paremeters
* **dynamic slicing** for massive memory savings and parallelism
* support for **hyper** edge tensor networks and thus arbitrary einsum equations
* **paths** that can be supplied to [`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), [`opt_einsum`](https://dgasmith.github.io/opt_einsum/), [`quimb`](https://quimb.readthedocs.io/en/latest/) among others
* **performing contractions** with tensors from many libraries via [`autoray`](https://github.com/jcmgray/autoray),
  even if they don't provide `einsum` or `tensordot` but do have (batch) matrix
  multiplication

<p align="center"><img src="https://imgur.com/jMO138y.png" alt="cotengra" width="500px"></p>
