"# API_DeepLearning" 

## 1. Preparación del entorno
    $ conda create -n APIDeep anaconda python=3.7.7
    $ conda activate APIDeep
    $ conda install ipykernel
    $ python -m ipykernel install --user --name APIDeep --display-name "APIDeep"
    $ conda install tensorflow-gpu==2.1.0 cudatoolkit=10.1
    $ pip install tensorflow==2.1.0
    $ pip install jupyter
    $ pip install keras==2.3.1
    $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all]
    
 ## 2. Entrenar la red neuronal
 
    Descargar el repositorio
    Abrir terminal en la carpeta y correr jupyter notebook
    
    $ jupyter notebook
    
    Ejecutar BirdClass.ipynb
    
 ## 3. Probar la red neuronal
 
    $ python TestModel.py
    
## 4. Probar el API de Flask

    $ python app.py

## Resultado

![Aquí la descripción de la imagen por si no carga](https://github.com/DavidReveloLuna/API_DeepLearning/blob/master/asssets/Resultado.jpg)
