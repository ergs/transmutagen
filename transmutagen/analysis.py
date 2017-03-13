import tables

def analyze(file):
    with tables.open_file(file, mode='r') as h5file:
        for lib in h5file.root:
            print(lib)
            table = h5file.get_node(lib, '/origen')
            for row in table:
                print(row['execution time ORIGEN'])

if __name__ == '__main__':
    import sys
    analyze(sys.argv[-1])
