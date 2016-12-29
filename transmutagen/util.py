import difflib

def diff_strs(a, b):
    """
    Print a colored character-by-character diff of a and b.

    Requires colorama to be installed.

    Differing characters from the a string are printed in red and those from
    the b string are printed in green.

    Note that this can be slow for large strings (more than 10000 characters).

    """
    try:
        import colorama
    except ImportError:
        raise ImportError("colorama is required to use diff_strs")

    s = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i1, j1, i2, j2 in s.get_opcodes():
        if op == 'equal':
            print(a[i1:j1], end='')
        elif op == 'replace':
            print(colorama.Fore.RED, a[i1:j1], colorama.Fore.GREEN, b[i2:j2], sep='', end='')
        elif op == 'insert':
            print(colorama.Fore.GREEN, b[i2:j2], sep='', end='')
        elif op == 'delete':
            print(colorama.Fore.RED, a[i1:j1], sep='', end='')
        print(colorama.Style.RESET_ALL, end='')
    print()