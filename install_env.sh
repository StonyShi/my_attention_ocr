

conda create -n test python=3.6 ipykernel
source activate test

python -m ipykernel install --user --name test --display-name "test"



#pip install -r requirements.txt

while read line ; do $line; done < requirements.txt