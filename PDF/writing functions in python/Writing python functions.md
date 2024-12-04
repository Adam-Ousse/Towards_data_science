docstring :
```python
def function():
	"""
	function documentation
	in google style : concise imperative description
	Args : 
		arg_1 (str): ...
		...
	Returns : 
		bool : ..
	Raises : 
		ValueError : ..
	Notes :
		...
	"""
```
to access the docstring : `function.__doc__`
or using the `inspect` module 
```python
import inspect
print(inspect.getdoc(funciton))
```