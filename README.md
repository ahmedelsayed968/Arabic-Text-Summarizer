# Arabic-Text-Summarizer

## Results
### Decoder-ONLY Models
- inceptionai/Jais-family-256m
  ![img](/assets/decoder-outputs.png)
  ![img](/assets/W&B%20Chart%2010_10_2024,%2007_14_57.png)
  ![img](/assets/W&B%20Chart%2010_10_2024,%2007_16_15.png)
  ![img](/assets/W&B%20Chart%2010_10_2024,%2007_18_26.png)
  | Metric   | Score                  |
  |----------|------------------------|
  | rouge1   | 0.024213605715402403   |
  | rouge2   | 0.0014741946283852877  |
  | rougeL   | 0.024084952075629662   |
  | rougeLsum| 0.02407977715402647    |


## Project Organization


├── README.md         		<- top-level README for developers using this project.    
├── pyproject.toml         		<- black code formatting configurations.    
├── .dockerignore         		<- Files to be ognored in docker image creation.    
├── .gitignore         		<- Files to be ignored in git check in.    
├── .pre-commit-config.yaml         		<- Things to check before git commit.    
├── .circleci/config.yml         		<- Circleci configurations       
├── .pylintrc         		<- Pylint code linting configurations.    
├── Dockerfile         		<- A file to create docker image.    
├── environment.yml 	    <- stores all the dependencies of this project    
├── main.py 	    <- A main file to run API server.    
├── src                     <- Source code files to be used by project.    
│       ├── inference 	        <- model output generator code   
│       ├── model	        <- model files   
│       ├── training 	        <- model training code  
│       ├── utility	        <- contains utility  and constant modules.   
├── logs                    <- log file path   
├── config                  <- config file path   
├── data              <- datasets files   
├── docs               <- documents from requirement,team collabaroation etc.   
├── tests               <- unit and performancetest cases files.   
│       ├── cov_html 	        <- Unit test cases coverage report    

## Installation
Development Environment used to create this project:  
Operating System: Windows 10 Home  

### Softwares
Anaconda:4.8.5  <a href="https://docs.anaconda.com/anaconda/install/windows/">Anaconda installation</a>   
 

### Python libraries:
Go to location of environment.yml file and run:  
```
conda env create -f environment.yml
```

 

## Usage
Here we have created ML inference on FastAPI server with dummy model output.

1. Go inside 'Arabic-Text-Summarizer' folder on command line.  
2. Run:
  ``` 
      conda activate Arabic-Text-Summarizer  
      python main.py       
  ```
3. Open 'http://localhost:5000/docs' in a browser.
   
 
### Unit Testing
1. Go inside 'tests' folder on command line.
2. Run:
  ``` 
      pytest -vv 
      pytest --cov-report html:tests/cov_html --cov=src tests/ 
  ```
 
### Performance Testing
1. Open 2 terminals and start main application in one terminal  
  ``` 
      python main.py 
  ```

2. In second terminal,Go inside 'tests' folder on command line.
3. Run:
  ``` 
      locust -f locust_test.py  
  ```

### Black- Code formatter
1. Go inside 'Arabic-Text-Summarizer' folder on command line.
2. Run:
  ``` 
      black src 
  ```

### Pylint -  Code Linting
1. Go inside 'Arabic-Text-Summarizer' folder on command line.
2. Run:
  ``` 
      pylint src  
  ```

### Containerization
1. Go inside 'Arabic-Text-Summarizer' folder on command line.
2. Run:
  ``` 
      docker build -t myimage .  
      docker run -d --name mycontainer -p 5000:5000 myimage         
  ```

### Pre-commit hooks
1. Go inside 'Arabic-Text-Summarizer' folder on command line.
2. Run:
  ``` 
      pre-commit install  
  ```
3. Whenever the command git commit is run, the pre-commit hooks will automatically be applied.     
4. To test before commit,run:  

  ``` 
      pre-commit  run 
  ```    

### CI/CD using Circleci
1. Add project on circleci website then monitor build on every commit.


## Contributing
Please create a Pull request for any change. 

## License


NOTE: This software depends on other packages that are licensed under different open source licenses.

