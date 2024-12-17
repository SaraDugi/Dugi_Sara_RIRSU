import json
from datetime import datetime, timedelta

data = []
start_date = datetime(2024, 6, 1)
value = 10

for i in range(187):
    date = start_date + timedelta(days=i)
    data.append({
        "date": date.strftime("%Y-%m-%d"),
        "available_bike_stands": value
    })
    value += 5  

# Save to JSON
with open("data.json", "w") as file:
    json.dump(data, file, indent=2)