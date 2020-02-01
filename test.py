import os, operator, sys
dirpath = os.path.abspath(sys.argv[1])
# make a generator for all file paths within dirpath
all_files = ( os.path.join(basedir, filename) for basedir, dirs, files in os.walk(dirpath) for filename in files   )

sorted_files = sorted(all_files, key = os.path.getsize)

# make a generator for tuples of file path and size: ('/Path/to/the.file', 1024)
files_and_sizes = ( (path, os.path.getsize(path)) for path in all_files )
sorted_files_with_size = sorted( files_and_sizes, key = operator.itemgetter(1) )
for f in sorted_files:
    print(os.path.getsize(f))