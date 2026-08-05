## Descripción
El algoritmo de optimización de políticas multimodales (MUPO) se desarrolla sobre el marco General Optimal control Problem Solver (GOPS) por el Intelligent Driving Laboratory (iDLab).
El repositorio contiene todos los entornos, algoritmos incluyendo líneas base, pesos de redes entrenados y datos utilizados para generar las gráficas como se describe en el artículo.

## Instalación
GOPS requiere:
1. Windows 7 o superior o Linux (Linux recomendado).
2. Python 3.6 o superior. Recomendamos usar Python 3.8.
3. La ruta de instalación debe estar en inglés.
4. PyTorch con soporte CUDA. Visite https://pytorch.org/ e instale la versión apropiada de PyTorch para su sistema, asegurándose de que coincida con su versión de CUDA.

Puede instalar GOPS a través de los siguientes pasos:
```bash
# crear entorno conda
cd code/
conda install mamba -n base -c conda-forge
mamba env create -f gops_environment.yml
conda activate gops
# instalar GOPS
pip install -e .
```

## Inicio Rápido
Este es un ejemplo de ejecución de MUPO en un entorno de seguimiento de vehículos y evasión de obstáculos llamado "bypass".
Entrene la política ejecutando:
```bash
python example_train/mupo_bypass.py
```
Después del entrenamiento, pruebe la política ejecutando:
```bash
python example_run/run_bypass.py
```
Luego puede ver los resultados de la simulación en la carpeta `figures`.
Recuerde modificar la ruta del modelo en `run_bypass.py`, incluyendo `algorithm`, `date_list`, `iteration`, y si desea usar dinámicas de lazo cerrado (`is_closed_loop`).
Cuando `is_closed_loop` se establece en False, se utiliza un modelo de dinámica de lazo abierto, y el estado del vehículo ego se actualiza según el estado predefinido en `step_open_loop()` del entorno.
Las figuras 4c y 4d del artículo utilizan el modelo de dinámica de lazo abierto, mientras que otros experimentos utilizan el modelo de dinámica de lazo cerrado.

Se proporcionan dos ejemplos de políticas continuas con colisiones
```bash
python run_bypass_collision.py
python run_encounter_collision.py
```

## Reconocimiento
Queremos agradecer a todos los miembros del Intelligent Driving Laboratory (iDLab), School of Vehicle and Mobility, Tsinghua University por sus excelentes contribuciones y por proporcionar consejos útiles para GOPS.
