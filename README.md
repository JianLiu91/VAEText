# VAEText

VAE for mnist and texts

## Main Requirements
1. Python3
2. Pytorch 0.4
3. Matplotlib


### VAE for mnist
`python main_vae.py`

Experimental results are in:
1. `out/_true` ground-truth examples
2. `out/_mean` reconstruct by mean
3. `out/_rec1` reconstruct by sample1
4. `out/_rec2` reconstruct by sample2
5. `out/_rand` random sampling from `z ~ N(0, 1)`


### VAE for text
`python main_sentence_vae.py`

Experimental results:
1. random sampling from `z ~ N(0,1)`
2. reconstruct by mean
3. reconstruct by sample1
4. reconstruct by sample2
5. interpolating from two sentences