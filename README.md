# prerequisites 
pillow and tensorflow  
you can install them with `python -m pip install pillow tensorflow numpy packaging`

# training with gpu
1. install [miniconda](https://docs.conda.io/en/latest/miniconda.html), and do `conda init --all`
3. create miniconda env: `conda create --name ai`, and activate it: `conda activate ai` (you will have to activate the environment every time you open your terminal)
4. install required modules: `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 nvidia cuda-nvcc`
5. install prerequisites: `python -m pip install pillow tensorflow numpy packaging`  
