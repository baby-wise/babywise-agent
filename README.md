# BabyWise_Agents
## Guía para el desarrollo del agent

### Inicializacion del entorno
1. Descargar el cliente de livkit en local: `winget install LiveKit.LiveKitCLI`
2. Agregar al root del proyecto el archivo `livekit.toml`
3. Abrir el navegador con la cuenta de BabyWise
4. En el root del proyecto abrir la consola: `lk cloud auth`
5. Se te va a abrir livekit y selecciona el proyecto que tenemos, solo hay uno
6. En la cosnola te va a decir nombre del proyecto y si es el default, yo le puse babywise y que sea el default

#### Opcional (para que no tire errores a la hora de escribir codigo)
* Crear un entorno virtual de python: `python -m venv venv`
* Activar el entrono virtual: `.venv\Scripts\activate`
* Instalar dependencias: `pip install -r requirements.txt`

### Deploy del Agent
* Asegurate de no estar en el entorno virtual de python, para salir usar: `deactivate`
* Para publicar una nueva version: `lk agent deploy`
* Para ver los logs del agent actual: `lk agent logs`
