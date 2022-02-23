from data import processor
from data import config


file_names = [config.FILE_NAME1, config.FILE_NAME2, config.FILE_NAME3]

new_file_names = [config.NEW_FILE_NAME1, config.NEW_FILE_NAME2, config.NEW_FILE_NAME3]

up_file_names = [config.UP_FILE_NAME1, config.UP_FILE_NAME2, config.UP_FILE_NAME3]

print(file_names)
print(new_file_names)
print(up_file_names)


for i, (fn, nfn, ufn) in enumerate(zip(file_names, new_file_names, up_file_names)):
    print(i, "\t", fn, "\t", nfn, "\t", ufn)
    processor.process_data(fn, nfn, ufn)





