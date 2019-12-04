import nbformat

def execute_notebook():
	try:
		from nbconvert.preprocessors import ExecutePreprocessor
		# with open('proposal.ipynb') as f:
		with open('Final Project.ipynb') as f:
		    nb = nbformat.read(f, as_version=4)
		ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
		ep.preprocess(nb, {'metadata': {'path': ''}})
		return 0
	except:
		return 1

execute_notebook()