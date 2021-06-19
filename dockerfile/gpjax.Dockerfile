FROM tensorflow/tensorflow

#install git
#RUN pip install --upgrade pip

#run requirements in order
#required because scipy needs numpy to already be installed. see https://stackoverflow.com/questions/51399515/docker-cannot-build-scipy.
COPY requirements.txt .
#hack for now untill model_log can be pip installed
RUN while read module; do pip3 install $module; done < requirements.txt

#install git
RUN apt-get update  && apt-get install -y git

# Create a working directory
RUN mkdir /app
WORKDIR /app
