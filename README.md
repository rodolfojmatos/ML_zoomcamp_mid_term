**Adult income binary classification**

The dataset can be find here from UC Irvine Repository\
https://archive.ics.uci.edu/dataset/2/adult

It comes in 2 separate dataset, one for train and another for test.\
I combined both in a big dataset.

There are 14 features:\
**age:** continuous.\
**workclass:** Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.\
**fnlwgt:** continuous.\
**education:** Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.\
**education-num:** continuous.\
**marital-status:** Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.\
**occupation:** Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, \Priv-house-serv, Protective-serv, Armed-Forces.\
**relationship:** Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.\
**race:** White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.\
**sex:** Female, Male.\
**capital-gain:** continuous.\
**capital-loss:** continuous.\
**hours-per-week:** continuous.\
**native-country:** United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy,\ Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia,\ El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

**Target variable:** income

The income will be mapped as '<=50k' as 0 and '>50k' as 1

The objective is to predict if a person's income is <=50k or >50K.

**Notebook.ipynb** - Data preparation / EDA and feature importance analysis / Model Selection proce and parameter tunning

**train.py** - Training the final model / Saving it to a file using pickle

**predict.py** - Loading the model / Serving it via Flask or waitress or gunicorn

**testing-predict.py** - run it to test the service that should be running with docker, flask, waitress or gunicorn

**Deploying with waitress:**
- Load both pipfile and pipfile.lock
- Run the following in Powershell/Command Prompt/Gitbash: pipenv run waitress-serve --listen=0.0.0.0:9696 predict:app
- Run the testing-predict.py file in another window of Powershell/Command Prompt/Gitbash

**Deploying with Docker:**
- Make sure the docker is running
- Create an image based on Dockerfile on Powershell/Command Prompt/Gitbash: docker build -t zoomcamp-mid .
- Run the image just created: docker run -it --rm -p 9696:9696 zoomcamp-mid
- Run the testing-predict.py file in another window of Powershell/Command Prompt/Gitbash

