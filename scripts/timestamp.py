import datetime

# Sample timestamps in microseconds
timestamps = [
    7090906, 7190913, 7290907, 7391908, 7490907, 7590907, 7690907, 7790908,
    7890913, 7990907, 8091938, 8190908, 8290907, 8390907, 8490907, 8590909,
    8690907, 8791909, 8890907, 8990907, 9091907, 9190907, 9290922, 9390907,
    9490916, 9590921, 9690926, 9790933, 9891907, 9990925, 10090907, 10191909,
    10291937, 10391935, 10491949, 10591948, 10691944
]

# Convert timestamps to human-readable date and time format
print("Timestamps in human-readable format:")
for timestamp in timestamps:
    dt = datetime.datetime.fromtimestamp(timestamp / 1000000.0)
    print(dt)
