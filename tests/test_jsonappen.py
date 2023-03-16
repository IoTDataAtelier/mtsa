
import json


# Open a file with access mode 'a'
with open("sample.txt", "a") as file_object:
    # Append 'hello' at the end of file
    file_object.write(json.dumps(result_fit['time_elapsed']))
