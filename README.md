# A generic and flexible regularization framework for NeRFs

> [!NOTE]
> This code is provided as is and will not be maintained. 
> Only the depth regularization is provided in this repo given the poor performance of the normals regularization process.

This repo contains the code used to generate the results presented in "A generic and flexible regularization framework for NeRFs", WACV 2024.

## RegNeRF

This repo is based on the code of [RegNeRF](https://github.com/google-research/google-research/tree/master/regnerf) to make the comparison easier.
Parameters and training configuration are (except for the regularization parameter lambda and the gradient clipping parameter gmax (Eq. 5) that are specific to this method) exactly the same as in RegNeRF.
Don't hesitate to check it out if you're interested in other regularization options!

You can also refer to it for other advice (e.g. data configuration) on running the code since the base is the same.

## Requirements and installation

This code is based on JAX. It has been tested with `jax==3.10` and the following jaxlib
```
pip install jaxlib==0.3.0+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

> [!WARNING]
> Old version of JAX were compiled for specific CUDA and CuDNN versions. Make sure that your system is properly configured.
> Given that this repo is based on an outdated JAX version, a docker file is also provided. Note that this docker configuration was not used to generate the results of the paper and therefore has not been intensively tested. It might contain some bugs.

Dependencies can then be installed using:
```
pip install -r requirements.txt
```

## Running the code

### Training a new model

Training a new model can be done using 
```
python train.py --gin_configs configs/{CONFIG}
```
where `{CONFIG}` is the requested config file.

Diffnerf training configurations are provided in `configs/` and contain the postfix `_diffnerf`. The other configuration files are from RegNeRF to compare the performance.

More infos in how to train a model can be found in [RegNeRF training description](https://github.com/google-research/google-research/tree/master/regnerf#training-an-new-model)

### Rendering test images

You can render and evaluate test images by running

```python eval.py --gin_configs configs/{CONFIG} ```

### Using a pre-trained model

Pre-trained models are provided as releases. They can be found in the [Release section](https://github.com/tehret/diffnerf/releases/tag/v1.0.0).

## Citations

If you find this work useful, please cite it as
```
@inproceedings{ehret2024generic,
  title={A generic and flexible regularization framework for NeRFs},
  author={Ehret, Thibaud and Mar{\'\i}, Roger and Facciolo, Gabriele},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3088--3097},
  year={2024}
}
```

Don't hesitate to also cite the work of RegNeRF, from which the large majority of this code comes from
```
@InProceedings{Niemeyer2021Regnerf,
    author    = {Michael Niemeyer and Jonathan T. Barron and Ben Mildenhall and Mehdi S. M. Sajjadi and Andreas Geiger and Noha Radwan},  
    title     = {RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs},
    booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
}
```

