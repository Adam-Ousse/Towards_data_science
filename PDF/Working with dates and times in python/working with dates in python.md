
`datetime.date object` 
```python
from datetime import date
Date = date(2003,12,28)
date.weekday() #number of weekday 0 for monday 6 for sunday
date.year
date.month
date.day
```
`datetime.timedelta`
```python
from datetime import timedelta
td = date(2003,12,28) - date(2002,12,28)
td.days #365
td.total_seconds()#total secs
td_1 = timedelta(days=200)
date_1 += td_1
```
Date formats
```python
#ISO 8601
print(date.isoformat())
#custom formats
print(date.strftime("%Y-%d-%m"))
date.strftime("%B")# for the month name
date.strftime("%j") # for day number 1-365
datetime.strftime(string, fmt)
```
`datetime.datetime`
```python
from datetime import datetime
date_time = datetime(2017,12,24,20,20,30,6999) #2017-12-24 , at 20:20:30:6999 microseconds
date_time.replace(year=2003, microsecond= 500)
date_time.strftime("%H:%M:%S") # for hour:minutes:seconds
```
getting date from UNIX timestamps which represents the seconds since jan 1st 1970
```python
from datetime import datetime
date = datetime.fromtimestamp(12412421321)

```
deltatime can also take seconds and microseconds , subtracting or adding dates returns a timedelta object 
`datetime.timezone` for timezone management
```python
from datetime import timezone
utc = timezone.utc
tunisian = timezone(timedelta(hours=1)) #utc +1

#creating a time in uk 
time = datetime(2020,12,30,12,05,20,tzinfo=utc)
#same time in tunisia
time_tunisia = time.astimezone(tunisian)
```
Time zone database `dateutil.tz`
```python
from dateutil import tz
Tunisia_time_zone = tz.gettz("Africa/Tunis")
Paris = tz.gettz("Europe/Tunis")
now = datetime.now()
now_paris = now.astimezone(Paris)
```

In winter we roll back to the standard time meaning 
```python
amb = datetime(2017,11,5,1,0,0,tzinfo=tz.gettz("US/Eastern"))
amb2 = datetime(2017,11,5,1,0,0,tzinfo=tz.gettz("US/Eastern"))
print(tz.datetime_ambiguous(amb2))# True 
print(amb)
amb2=tz.enfold(amb2)
print(amb2)
print(amb2-amb)
```
Key points covered:

	- **Ambiguous Times**: You learned that when clocks fall back, there are two instances of the same local time. For example, 1 AM appears twice.
	- **UTC Conversion**: To handle these ambiguities, you should convert local times to Coordinated Universal Time (UTC), which is unambiguous.
	- **Using `tz.enfold()`**: This method marks which instance of the ambiguous time you are referring to. For example, `tz.enfold(datetime)` specifies the second occurrence of the time.
	
	Here's a code snippet to identify ambiguous start and end times in your bike trip data:
	
	```
	# Loop over trips
	for trip in onebike_datetimes:
	  # Rides with ambiguous start
	  if tz.datetime_ambiguous(trip['start']):
	    print("Ambiguous start at " + str(trip['start']))
	  # Rides with ambiguous end
	  if tz.datetime_ambiguous(trip['end']):
	    print("Ambiguous end at " + str(trip['end']))
	```
	
	By converting to UTC and using `tz.enfold()`, you ensure accurate time calculations across daylight saving boundaries.
	
	The goal of the next lesson is to teach how to handle time zones and daylight saving in Python by making datetime objects timezone-aware for accurate global time comparisons

pandas datetime
```python
df = pdf.read_csv(file, parse_dates= ["lol"])
df["date"]= pd.to_datetime(df["date"], format = "%Y/%m/%d %H:%M:%S")
df["duration"].dt.total_seconds()
```
We can apply mean, median and sum for timedeltas.
we can analyze data over different periods using `.resample("M", on="start date")` (over months)
Setting timezone : 
```python
rides["date"] = rides["date"].dt.tz_localize("America/New_York",ambiguous= "NaT") #instead of raising an error it sets ambiguous dates to NaT.
rides["date_london"] = rides["date"].dt.tz_convert("Europe/London")
```
`df["col"].dt.day_name()`
shifting rows:
```python
rides["end date"].shift(1) #shifts the rows by 1
```
