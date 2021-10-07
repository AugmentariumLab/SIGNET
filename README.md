# SIGNET: Efficient Neural Representations for Light Fields
This repository contains the training and inference code for SIGNET: Efficient Neural Representations for Light Fields, published at ICCV 2021.

## Requirements
* CUDA
* PyTorch
* Numpy
* PIL

## Demo

To decode an image at light field view point (u, v), please run
* `python demo_decode.py -u [u] -v [v] --scene [scene_name]`
u and v are integers within the range [0, 16]. 
For demo we provide the pretrained weights for scenes "lego" and "tarot" in the `encoded_weights` folder.

## Related Publication

Please refer to <https://augmentariumlab.github.io/SIGNET> for our paper published in ICCV 2021: "SIGNET: Efficient Neural Representations for Light Fields".

## References

If you use this in your research, please reference it as:

    @inproceedings{Feng2021SIGNET,
      author={Feng, Brandon Y. and Varshney, Amitabh},
      booktitle={Proceedings of the International Conference on Computer Vision (ICCV 2021)},
      title={SIGNET: Efficient Neural Representations for Light Fields},
      year={2021},
    }

or

    Brandon Y, Feng and Amitabh Varshney. 2021. SIGNET: Efficient Neural Representations for Light Fields. International Conference on Computer Vision (ICCV 2021).
