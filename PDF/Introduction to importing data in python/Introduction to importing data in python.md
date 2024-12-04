- Reading text files 
	- `file = open(filename, mode="r`) for reading, 
	- `text= file.read()` to get the content of the file
	- `file.readline()` gets a line, recalling the method returns the next line
	- `file.close()` to close the connection to the file
	- `with open(file,ame,"r") as file : ` best practice because you don't need to remind yourself to close the file.
- Reading flat files with numeric values to numpy
	- `np.loadtst(filename, delimiter=",",skiprows=1,usecols[0,4],dtype=str)` delimiter by default " " , skiprows 1 if the header is string, uses the first and 4th col
- Reading using pandas :
	- `df =pd.read_csv(filename, nrows=n, header=None)` header= None if there s no header, it will be indexed from 0 to n 
		- `sep` sets the expected delimiter.
		    - You can use `','` for comma-delimited.
		    - You can use `'\t'` for tab-delimited.
		- `comment` takes characters that comments occur after in the file.
		- `na_values` takes a list of strings to recognize as `NA`/`NaN`. By default, some values are already recognized as `NA`/`NaN`. Providing this argument will supply additional values.
	- `df.to_numpy()` to turn it into a numpy array
- Other file formats (excel spreadsheets, matlab files, sas, stata, HDF5, pickled)
	- Pickle :
		- after importing pickle `with open("file.pkl","rb") as file :` `data = pickle.load(file)`
	- Excel: .xlsx
		- `data=pd.ExcelFile(file)
		- `data.sheet_names` for different sheets
		- `df1 = data.parse("sheet_name_1)` or `data.parse(0)` for first sheet
			- `skiprows=` either interger or a list of rows
			- `names=[]` list of new columns names
			- `usecols=[]` list of columns to use
		- or using `pd.read_excel(file,sheet_name=None)` sheet_name = None to import all sheets which returns a dict sheetname => df sheet
	- SAS/Stat files .sas7bdat and .sas7bcat/ .dta
		```python
		from sas7bdat import SAS7BDAT
		with SAS7BDAT("data.sas7bdat") as f:
			df_sas = f.to_data_frame()
		```
	- `data = pd.read_stata("data.dta")` for stata
	- HDF5 .hdf5 
		```python
		import h5py
		data = h5py.File(filename,"r")
		#it's like a dictionary .keys() etc.. {"meta":,"quality":,"stratin":..}
		```
	- Matlab files .mat
		```python
		import scipy.io
		mat = scipy.io.loadmat(filename) #dictionary key : variable names , value : variables values
		```
- Relation databases , each table represents an entity, each row is an instance of said entity like orders, employees, companies etc.. (Codd's 12 rules describe RDM systems like postgres, mysql and sqlite)
	- Connecting to a database / creating a database engine 
		- sqlite3
		- SQLAlchemy works with many relation database management systems
			```python
			from sqlalchemy import create_engine
			engine = create_engine("sqlite:///data.sqlite") # type of database/// name which is called a connection string
			```
			- `engine.table_names()` to get the table names
	- Querying relational databases 
		- sqlalchemy
			- `SELECT * FROM Table_Name WHERE condition` returns all column from the table_name where the condition is satisifed is an sql command 
			- `ORDER BY column_name` to order the records
			```python
			con = engine.connect() #or do with engine.connect() as con : ...
			rs = con.execute("SELECT * FROM Orders")
			df= pd.DataFrame(rs.fetchall()) #or rs.fetchmany(size=100)
			df.columns = rs.keys()
			
			con.close()
			
			```
	- pandas :
		- `df = pd.read_sql_query("Select* from orders",engine)`
		- `df = pd.read_sql_query("select oderid , companyname from orders inner join customers on orders.customerid = customers.customerid",engine)`
		```sql
		SELECT books.id AS book_id, authors.name AS author_name
		FROM books
		INNER JOIN authors ON books.author_id = authors.id;
		```