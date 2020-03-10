# Unsupervised image matching and object discovery as optimization

Huy V. Vo, Francis Bach, Minsu Cho, Kai Han, Yann LeCun, Patrick Pérez, Jean Ponce - CVPR 2019


### Getting started

The code is written and tested with Matlab 2017a. Modifications might be necessary to run it with other versions of Matlab.

### Installing 

```
git clone https://github.com/vohuy93/OSD.git
```

```
cd OSD
```

### Dependencies

You need to download the code for generating region proposals with randomized Prim's algorithm and put it in the UODOptim/ folder

```
git clone https://github.com/smanenfr/rp.git
```

then run

```
cd rp; matlab -r "setup"
```

## Running the tests

The main script for testing the code on VOC_6x2 is scripts/run_UOD.m. In a terminal, from the UODOptim folder, run 

```
cd scripts; matlab -r "run_OSD"
```

## Citations

```
@INPROCEEDINGS{Vo19UOD,
  title     = {Unsupervised image matching and object discovery as optimization},
  author    = {Vo, Huy V. and Bach, Francis and Cho, Minsu and Han, Kai and LeCun, Yann and P{\'e}rez, Patrick and Ponce, Jean},
  booktitle = {CVPR},
  year      = {2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The code for Probabilistic Hough Matching (PHM) algorithm is taken from the [project page](https://www.di.ens.fr/willow/research/objectdiscovery/) of the paper "Unsupervised Object Discovery and Localization in the Wild".

This work was supported in part by the Inria/NYU collaboration agreement, the Louis Vuitton/ENS chair on artificial intellgence and the EPSRC Programme Grant Seebibyte EP/M013774/1.
