Write-Output "Setting up Python environment for ML and DS"
python -m venv test-env
 
Write-Output "Importing libraries"
python -m pip install pandas
python -m pip install matplotlib
python -m pip install numpy
python -m pip install -U scikit-learn